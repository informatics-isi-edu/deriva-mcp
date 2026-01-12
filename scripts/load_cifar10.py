#!/usr/bin/env python3
"""Load CIFAR-10 dataset into DerivaML catalog via MCP.

This script downloads the CIFAR-10 dataset from Kaggle and loads it into
a Deriva catalog using the DerivaML MCP server. It creates:
- An Image asset table for storing image files
- An Image_Class vocabulary with the 10 CIFAR-10 classes
- An Image_Classification feature linking images to their class labels
- A dataset hierarchy: Complete (all images), Segmented (Training + Testing)

Usage:
    python load_cifar10.py --hostname ml.derivacloud.org --catalog-id 99

Requirements:
    - Kaggle CLI configured (~/.kaggle/kaggle.json)
    - deriva-ml-mcp server available
    - mcp Python package installed
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import subprocess
import sys
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# CIFAR-10 class definitions
CIFAR10_CLASSES = [
    ("airplane", "Fixed-wing aircraft", ["plane", "aeroplane"]),
    ("automobile", "Motor vehicle with four wheels", ["car", "auto"]),
    ("bird", "Feathered flying vertebrate", []),
    ("cat", "Small domestic feline", ["kitten"]),
    ("deer", "Hoofed ruminant mammal", []),
    ("dog", "Domestic canine", ["puppy"]),
    ("frog", "Tailless amphibian", ["toad"]),
    ("horse", "Large domesticated hoofed mammal", ["pony"]),
    ("ship", "Large watercraft", ["boat", "vessel"]),
    ("truck", "Motor vehicle for transporting cargo", ["lorry"]),
]


class MCPClient:
    """Client for calling DerivaML MCP tools."""

    def __init__(self, session: ClientSession):
        self.session = session

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call an MCP tool and return parsed JSON result."""
        result = await self.session.call_tool(name, arguments=arguments or {})
        if result.content and len(result.content) > 0:
            return json.loads(result.content[0].text)
        return {}

    # Connection tools
    async def connect_catalog(
        self, hostname: str, catalog_id: str, domain_schema: str | None = None
    ) -> dict[str, Any]:
        args = {"hostname": hostname, "catalog_id": catalog_id}
        if domain_schema:
            args["domain_schema"] = domain_schema
        return await self.call_tool("connect_catalog", args)

    # Vocabulary tools
    async def list_vocabularies(self) -> list[dict[str, Any]]:
        result = await self.call_tool("list_vocabularies")
        return result if isinstance(result, list) else []

    async def create_vocabulary(self, vocabulary_name: str, comment: str = "") -> dict[str, Any]:
        return await self.call_tool(
            "create_vocabulary", {"vocabulary_name": vocabulary_name, "comment": comment}
        )

    async def add_term(
        self,
        vocabulary_name: str,
        term_name: str,
        description: str,
        synonyms: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "add_term",
            {
                "vocabulary_name": vocabulary_name,
                "term_name": term_name,
                "description": description,
                "synonyms": synonyms or [],
            },
        )

    async def list_vocabulary_terms(self, vocabulary_name: str) -> list[dict[str, Any]]:
        result = await self.call_tool("list_vocabulary_terms", {"vocabulary_name": vocabulary_name})
        return result if isinstance(result, list) else []

    # Schema tools
    async def list_tables(self) -> list[dict[str, Any]]:
        result = await self.call_tool("list_tables")
        return result if isinstance(result, list) else []

    async def create_asset_table(
        self,
        asset_name: str,
        columns: list[dict[str, Any]] | None = None,
        referenced_tables: list[str] | None = None,
        comment: str = "",
    ) -> dict[str, Any]:
        return await self.call_tool(
            "create_asset_table",
            {
                "asset_name": asset_name,
                "columns": columns,
                "referenced_tables": referenced_tables,
                "comment": comment,
            },
        )

    async def list_assets(self, asset_table: str) -> list[dict[str, Any]]:
        result = await self.call_tool("list_assets", {"asset_table": asset_table})
        return result if isinstance(result, list) else []

    # Feature tools
    async def create_feature(
        self,
        table_name: str,
        feature_name: str,
        comment: str = "",
        terms: list[str] | None = None,
        assets: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "create_feature",
            {
                "table_name": table_name,
                "feature_name": feature_name,
                "comment": comment,
                "terms": terms,
                "assets": assets,
            },
        )

    # Dataset tools
    async def create_dataset(
        self,
        description: str = "",
        dataset_types: list[str] | None = None,
        version: str | None = None,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "create_dataset",
            {"description": description, "dataset_types": dataset_types or [], "version": version},
        )

    async def add_dataset_members(self, dataset_rid: str, member_rids: list[str]) -> dict[str, Any]:
        return await self.call_tool(
            "add_dataset_members", {"dataset_rid": dataset_rid, "member_rids": member_rids}
        )

    async def add_dataset_element_type(self, table_name: str) -> dict[str, Any]:
        return await self.call_tool("add_dataset_element_type", {"table_name": table_name})

    # Workflow tools
    async def list_workflow_types(self) -> list[dict[str, Any]]:
        result = await self.call_tool("list_workflow_types")
        return result if isinstance(result, list) else []

    async def add_workflow_type(self, type_name: str, description: str) -> dict[str, Any]:
        return await self.call_tool(
            "add_workflow_type", {"type_name": type_name, "description": description}
        )

    # Execution tools
    async def create_execution(
        self,
        workflow_name: str,
        workflow_type: str,
        description: str = "",
        dataset_rids: list[str] | None = None,
        asset_rids: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "create_execution",
            {
                "workflow_name": workflow_name,
                "workflow_type": workflow_type,
                "description": description,
                "dataset_rids": dataset_rids or [],
                "asset_rids": asset_rids or [],
            },
        )

    async def start_execution(self) -> dict[str, Any]:
        return await self.call_tool("start_execution")

    async def stop_execution(self) -> dict[str, Any]:
        return await self.call_tool("stop_execution")

    async def get_execution_working_dir(self) -> dict[str, Any]:
        return await self.call_tool("get_execution_working_dir")

    async def upload_execution_outputs(self, clean_folder: bool = True) -> dict[str, Any]:
        return await self.call_tool("upload_execution_outputs", {"clean_folder": clean_folder})

    async def asset_file_path(
        self,
        asset_name: str,
        file_name: str,
        asset_types: list[str] | None = None,
        copy_file: bool = False,
        rename_file: str | None = None,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "asset_file_path",
            {
                "asset_name": asset_name,
                "file_name": file_name,
                "asset_types": asset_types,
                "copy_file": copy_file,
                "rename_file": rename_file,
            },
        )

    async def get_chaise_url(self, table_or_rid: str) -> dict[str, Any]:
        return await self.call_tool("get_chaise_url", {"table_or_rid": table_or_rid})

    async def resolve_rid(self, rid: str) -> dict[str, Any]:
        return await self.call_tool("resolve_rid", {"rid": rid})


@asynccontextmanager
async def create_mcp_client():
    """Create and initialize an MCP client connected to deriva-ml-mcp."""
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "deriva-ml-mcp"],
    )

    async with stdio_client(server_params) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            yield MCPClient(session)


def verify_kaggle_credentials() -> bool:
    """Check if Kaggle credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        logger.error(
            "Kaggle credentials not found. Please configure ~/.kaggle/kaggle.json\n"
            "See: https://www.kaggle.com/docs/api#authentication"
        )
        return False
    return True


def download_cifar10(temp_dir: Path) -> Path:
    """Download CIFAR-10 dataset from Kaggle.

    Returns:
        Path to the extracted dataset directory
    """
    download_dir = temp_dir / "cifar10"
    download_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading CIFAR-10 from Kaggle...")
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", "cifar-10", "-p", str(download_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Kaggle download failed: {result.stderr}")
        raise RuntimeError(f"Failed to download CIFAR-10: {result.stderr}")

    # Extract the outer zip file
    zip_files = list(download_dir.glob("*.zip"))
    if zip_files:
        logger.info("Extracting outer zip archive...")
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(download_dir)

    # CIFAR-10 from Kaggle uses 7z archives for train/test data
    # Extract 7z files using the 7z command (must be installed)
    seven_z_files = list(download_dir.glob("*.7z"))
    if seven_z_files:
        logger.info("Extracting 7z archives (train.7z, test.7z)...")
        for seven_z_file in seven_z_files:
            # Use 7z command to extract
            result = subprocess.run(
                ["7z", "x", str(seven_z_file), f"-o{download_dir}", "-y"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                # Try with 7za if 7z not found
                result = subprocess.run(
                    ["7za", "x", str(seven_z_file), f"-o{download_dir}", "-y"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Failed to extract {seven_z_file.name}. "
                        "Please install 7-zip: brew install p7zip"
                    )

    return download_dir


def load_train_labels(data_dir: Path) -> dict[str, str]:
    """Load training labels from trainLabels.csv.

    Returns:
        Dictionary mapping image ID to class label
    """
    labels = {}
    labels_file = data_dir / "trainLabels.csv"

    if labels_file.exists():
        with open(labels_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row["id"]] = row["label"]
    else:
        logger.warning("trainLabels.csv not found, will try to infer labels from directory structure")

    return labels


def iter_images(data_dir: Path, split: str, labels: dict[str, str]):
    """Iterate over images with their class labels.

    Yields:
        (image_path, class_name, image_id) tuples
    """
    if split == "train":
        train_dir = data_dir / "train"
        if train_dir.exists():
            for img_path in sorted(train_dir.glob("*.png")):
                image_id = img_path.stem
                class_name = labels.get(image_id)
                if class_name:
                    yield img_path, class_name, image_id
    else:
        # Test images - Kaggle test set doesn't have public labels
        # We'll skip test for now since labels aren't available
        test_dir = data_dir / "test"
        if test_dir.exists():
            for img_path in sorted(test_dir.glob("*.png")):
                image_id = img_path.stem
                # Test labels are not publicly available
                yield img_path, None, image_id


async def setup_domain_model(client: MCPClient) -> dict[str, Any]:
    """Create the domain model for CIFAR-10.

    Returns:
        Dictionary with created resource information
    """
    results = {}

    # Check existing vocabularies
    vocabs = await client.list_vocabularies()
    existing_vocabs = {v["name"] for v in vocabs}

    # Create Image_Class vocabulary if needed
    if "Image_Class" not in existing_vocabs:
        logger.info("Creating Image_Class vocabulary...")
        result = await client.create_vocabulary(
            vocabulary_name="Image_Class",
            comment="CIFAR-10 image classification categories",
        )
        results["vocabulary"] = result
    else:
        logger.info("Image_Class vocabulary already exists")
        results["vocabulary"] = {"status": "exists", "name": "Image_Class"}

    # Add class terms
    existing_terms = {t["name"] for t in await client.list_vocabulary_terms("Image_Class")}

    logger.info("Adding CIFAR-10 class terms...")
    term_results = []
    for class_name, description, synonyms in CIFAR10_CLASSES:
        if class_name not in existing_terms:
            result = await client.add_term(
                vocabulary_name="Image_Class",
                term_name=class_name,
                description=description,
                synonyms=synonyms,
            )
            term_results.append(result)
            logger.info(f"  Added term: {class_name}")
        else:
            logger.info(f"  Term exists: {class_name}")
    results["terms"] = term_results

    # Check existing tables
    tables = await client.list_tables()
    existing_tables = {t["name"] for t in tables}

    # Create Image asset table if needed
    # Note: We don't add Width/Height columns because all CIFAR-10 images are 32x32.
    # Adding columns to an asset table creates a directory structure for uploads
    # that requires those column values, which complicates the upload process.
    if "Image" not in existing_tables:
        logger.info("Creating Image asset table...")
        result = await client.create_asset_table(
            asset_name="Image",
            columns=[],  # No extra columns - just standard asset columns
            comment="CIFAR-10 32x32 RGB images",
        )
        results["asset_table"] = result
    else:
        logger.info("Image asset table already exists")
        results["asset_table"] = {"status": "exists", "table_name": "Image"}

    # Always ensure Image is registered as a dataset element type
    # (This is idempotent - safe to call even if already registered)
    logger.info("Enabling Image as dataset element type...")
    element_types = await client.call_tool("list_dataset_element_types")
    if "Image" not in element_types:
        await client.add_dataset_element_type("Image")

    # Create Image_Classification feature
    logger.info("Creating Image_Classification feature...")
    try:
        result = await client.create_feature(
            table_name="Image",
            feature_name="Image_Classification",
            comment="CIFAR-10 class label for each image",
            terms=["Image_Class"],
        )
        results["feature"] = result
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info("Image_Classification feature already exists")
            results["feature"] = {"status": "exists", "feature_name": "Image_Classification"}
        else:
            raise

    return results


async def setup_workflow_type(client: MCPClient) -> None:
    """Ensure Ingest workflow type exists."""
    workflow_types = await client.list_workflow_types()
    existing_types = {t["name"] for t in workflow_types}

    if "Ingest" not in existing_types:
        logger.info("Creating Ingest workflow type...")
        await client.add_workflow_type(
            type_name="Ingest",
            description="Data ingestion workflow for loading external datasets",
        )


async def setup_dataset_types(client: MCPClient) -> None:
    """Ensure required dataset types exist in Dataset_Type vocabulary."""
    logger.info("Setting up dataset types...")

    # Define the dataset types we need: (term_name, description, synonyms)
    required_types = [
        ("Complete", "A complete dataset containing all data", ["complete", "entire"]),
        ("Training", "A dataset subset used for model training", ["training", "train", "Train"]),
        ("Testing", "A dataset subset used for model testing/evaluation", ["test", "Test"]),
        ("Split", "A dataset that contains nested dataset splits", ["split"]),
    ]

    for type_name, description, synonyms in required_types:
        try:
            result = await client.add_term(
                vocabulary_name="Dataset_Type",
                term_name=type_name,
                description=description,
                synonyms=synonyms,
            )
            if result.get("status") == "error" and "already exists" not in result.get("message", "").lower():
                logger.warning(f"Could not add dataset type {type_name}: {result.get('message')}")
            else:
                logger.info(f"  Added dataset type: {type_name}")
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"Could not add dataset type {type_name}: {e}")


async def create_dataset_hierarchy(client: MCPClient) -> dict[str, str]:
    """Create the dataset hierarchy.

    Returns:
        Dictionary mapping dataset names to RIDs
    """
    datasets = {}

    logger.info("Creating dataset hierarchy...")

    # Create Complete dataset
    result = await client.create_dataset(
        description="Complete CIFAR-10 dataset with all labeled images",
        dataset_types=["Complete"],
    )
    if result.get("status") == "error":
        raise RuntimeError(f"Failed to create Complete dataset: {result.get('message')}")
    datasets["complete"] = result["rid"]
    logger.info(f"  Created Complete dataset: {result['rid']}")

    # Create Split dataset (contains nested Training/Testing)
    result = await client.create_dataset(
        description="CIFAR-10 dataset split into training and testing subsets",
        dataset_types=["Split"],
    )
    if result.get("status") == "error":
        raise RuntimeError(f"Failed to create Split dataset: {result.get('message')}")
    datasets["split"] = result["rid"]
    logger.info(f"  Created Split dataset: {result['rid']}")

    # Create Training dataset
    result = await client.create_dataset(
        description="CIFAR-10 training set with 50,000 labeled images",
        dataset_types=["Training"],
    )
    if result.get("status") == "error":
        raise RuntimeError(f"Failed to create Training dataset: {result.get('message')}")
    datasets["training"] = result["rid"]
    logger.info(f"  Created Training dataset: {result['rid']}")

    # Create Testing dataset
    result = await client.create_dataset(
        description="CIFAR-10 testing set",
        dataset_types=["Testing"],
    )
    if result.get("status") == "error":
        raise RuntimeError(f"Failed to create Testing dataset: {result.get('message')}")
    datasets["testing"] = result["rid"]
    logger.info(f"  Created Testing dataset: {result['rid']}")

    # Add Training and Testing as children of Split
    # Note: Use add_dataset_child for nested dataset relationships, not add_dataset_members
    await client.call_tool(
        "add_dataset_child",
        {"parent_rid": datasets["split"], "child_rid": datasets["training"]},
    )
    await client.call_tool(
        "add_dataset_child",
        {"parent_rid": datasets["split"], "child_rid": datasets["testing"]},
    )
    logger.info("  Linked Training and Testing to Split dataset")

    return datasets


async def load_images(
    client: MCPClient,
    data_dir: Path,
    datasets: dict[str, str],
    batch_size: int = 500,
    max_images: int | None = None,
) -> dict[str, Any]:
    """Load images into the catalog using execution system.

    Uses asset_file_path to register each image file for upload with the
    "Image" asset type, then calls upload_execution_outputs to upload all
    registered assets to the catalog.

    Args:
        client: MCP client instance
        data_dir: Path to extracted CIFAR-10 data
        datasets: Dictionary mapping dataset names to RIDs
        batch_size: Number of images per batch (for progress logging)
        max_images: Maximum number of images to upload (None = all images)

    Returns:
        Summary of loaded images
    """
    # Ensure Ingest workflow type exists
    await setup_workflow_type(client)

    # Create execution for provenance
    logger.info("Creating execution for data loading...")
    exec_result = await client.create_execution(
        workflow_name="CIFAR-10 Data Load",
        workflow_type="Ingest",
        description="Load CIFAR-10 dataset images into DerivaML catalog",
    )
    logger.info(f"  Execution RID: {exec_result.get('execution_rid')}")

    try:
        await client.start_execution()
        logger.info("  Execution started")

        # Load training labels
        labels = load_train_labels(data_dir)
        logger.info(f"Loaded {len(labels)} training labels")

        # Track images for dataset assignment
        train_images = []
        all_images = []

        # Process training images using asset_file_path
        limit_msg = f" (max: {max_images})" if max_images else ""
        logger.info(f"Registering training images for upload...{limit_msg}")
        count = 0
        for img_path, class_name, image_id in iter_images(data_dir, "train", labels):
            if class_name is None:
                continue

            # Check if we've reached the limit
            if max_images and count >= max_images:
                logger.info(f"  Reached limit of {max_images} images")
                break

            # Create unique filename with class prefix for tracking
            new_filename = f"{class_name}_{image_id}.png"

            # Register file for upload using asset_file_path
            # This copies the file and registers it with asset_type "Image"
            await client.asset_file_path(
                asset_name="Image",
                file_name=str(img_path),
                asset_types=["Image"],
                copy_file=True,
                rename_file=new_filename,
            )

            train_images.append(new_filename)
            all_images.append(new_filename)
            count += 1

            if count % 1000 == 0:
                logger.info(f"  Registered {count} images...")

        logger.info(f"  Total training images registered: {count}")

        # Stop execution before upload
        await client.stop_execution()
        logger.info("  Execution stopped")

        # Upload all registered assets to catalog
        logger.info("Uploading images to catalog...")
        upload_result = await client.upload_execution_outputs(clean_folder=True)
        logger.info(f"  Upload result: {upload_result}")

    except Exception as e:
        # Try to stop execution on error
        try:
            await client.stop_execution()
        except Exception:
            pass
        raise e

    # Get uploaded image RIDs and assign to datasets
    logger.info("Assigning images to datasets...")
    assets = await client.list_assets("Image")
    logger.info(f"  Found {len(assets)} uploaded images")

    if assets:
        # Get all RIDs
        all_rids = [a["RID"] for a in assets]

        # Add all images to Complete dataset
        logger.info("  Adding images to Complete dataset...")
        for i in range(0, len(all_rids), batch_size):
            batch = all_rids[i : i + batch_size]
            await client.add_dataset_members(datasets["complete"], batch)
        logger.info(f"    Added {len(all_rids)} images")

        # Add training images to Training dataset
        # For now, all images are training since test labels aren't public
        logger.info("  Adding images to Training dataset...")
        for i in range(0, len(all_rids), batch_size):
            batch = all_rids[i : i + batch_size]
            await client.add_dataset_members(datasets["training"], batch)
        logger.info(f"    Added {len(all_rids)} images")

    return {
        "total_images": len(all_images),
        "training_images": len(train_images),
        "uploaded_assets": len(assets),
    }


async def main_async(args: argparse.Namespace) -> int:
    """Main async entry point."""
    async with create_mcp_client() as client:
        # Either create a new catalog or connect to existing one
        if args.create_catalog:
            logger.info(f"Creating new catalog on {args.hostname} with project name: {args.create_catalog}")
            create_result = await client.call_tool(
                "create_catalog",
                {"hostname": args.hostname, "project_name": args.create_catalog},
            )

            if create_result.get("status") == "error":
                logger.error(f"Catalog creation failed: {create_result.get('message')}")
                return 1

            catalog_id = create_result["catalog_id"]
            domain_schema = create_result.get("domain_schema")

            # Print catalog ID prominently so user can find it
            print(f"\n{'='*60}")
            print(f"  CREATED NEW CATALOG")
            print(f"  Hostname:    {args.hostname}")
            print(f"  Catalog ID:  {catalog_id}")
            print(f"  Schema:      {domain_schema}")
            print(f"{'='*60}\n")

            logger.info(f"Created catalog {catalog_id}, domain schema: {domain_schema}")
        else:
            logger.info(f"Connecting to {args.hostname}, catalog {args.catalog_id}")
            conn_result = await client.connect_catalog(
                hostname=args.hostname,
                catalog_id=str(args.catalog_id),
                domain_schema=args.domain_schema,
            )

            if conn_result.get("status") == "error":
                logger.error(f"Connection failed: {conn_result.get('message')}")
                return 1

            catalog_id = args.catalog_id
            domain_schema = conn_result.get("domain_schema")
            logger.info(f"Connected to catalog, domain schema: {domain_schema}")

        # Set up domain model
        logger.info("Setting up domain model...")
        await setup_domain_model(client)
        logger.info("Domain model setup complete")

        # Apply catalog annotations for Chaise web interface
        logger.info("Applying catalog annotations...")
        project_name = args.create_catalog if args.create_catalog else domain_schema
        await client.call_tool(
            "apply_catalog_annotations",
            {
                "navbar_brand_text": f"CIFAR-10 ({project_name})",
                "head_title": "CIFAR-10 ML Catalog",
            },
        )

        # Setup dataset types and create dataset hierarchy
        await setup_dataset_types(client)
        datasets = await create_dataset_hierarchy(client)

        load_result = None
        if not args.dry_run:
            # Download CIFAR-10 from Kaggle
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                data_dir = download_cifar10(temp_path)
                logger.info(f"Downloaded CIFAR-10 to: {data_dir}")

                # Load images
                load_result = await load_images(
                    client, data_dir, datasets, args.batch_size, max_images=args.test
                )
                logger.info(f"Loading complete: {load_result}")
        else:
            logger.info("Dry run mode - skipping image download and upload")

        # Get Chaise URLs for datasets if requested
        dataset_urls = {}
        if args.show_urls:
            logger.info("Fetching Chaise URLs for datasets...")
            for name, rid in datasets.items():
                try:
                    url_result = await client.get_chaise_url(rid)
                    dataset_urls[name] = url_result.get("url", "")
                    logger.info(f"  {name}: {dataset_urls[name]}")
                except Exception as e:
                    logger.warning(f"  Failed to get URL for {name}: {e}")
                    dataset_urls[name] = ""

        # Print summary
        print("\n" + "=" * 60)
        print("  CIFAR-10 LOADING COMPLETE")
        print("=" * 60)
        print(f"  Hostname:      {args.hostname}")
        print(f"  Catalog ID:    {catalog_id}")
        print(f"  Schema:        {domain_schema}")
        print("")
        print("  Datasets created:")
        if args.show_urls and dataset_urls:
            print(f"    - Complete:   {datasets['complete']}")
            print(f"      URL: {dataset_urls.get('complete', 'N/A')}")
            print(f"    - Split:      {datasets['split']}")
            print(f"      URL: {dataset_urls.get('split', 'N/A')}")
            print(f"    - Training:   {datasets['training']}")
            print(f"      URL: {dataset_urls.get('training', 'N/A')}")
            print(f"    - Testing:    {datasets['testing']}")
            print(f"      URL: {dataset_urls.get('testing', 'N/A')}")
        else:
            print(f"    - Complete:   {datasets['complete']}")
            print(f"    - Split:      {datasets['split']}")
            print(f"    - Training:   {datasets['training']}")
            print(f"    - Testing:    {datasets['testing']}")
        if load_result:
            print("")
            print(f"  Images loaded: {load_result['total_images']}")
        if not args.show_urls:
            print("")
            print("  Tip: Use --show-urls to display Chaise URLs for each dataset")
        print("=" * 60 + "\n")

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load CIFAR-10 dataset into DerivaML catalog via MCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create a new catalog and load CIFAR-10
    python load_cifar10.py --hostname localhost --create-catalog cifar10_demo

    # Load into an existing catalog
    python load_cifar10.py --hostname ml.derivacloud.org --catalog-id 99

    # Dry run (create schema/datasets only)
    python load_cifar10.py --hostname localhost --create-catalog test --dry-run

    # Test mode (upload only 10 images)
    python load_cifar10.py --hostname localhost --create-catalog test --test

    # Test mode with custom limit (upload 100 images)
    python load_cifar10.py --hostname localhost --create-catalog test --test 100
        """,
    )
    parser.add_argument(
        "--hostname",
        required=True,
        help="Deriva server hostname (e.g., localhost, ml.derivacloud.org)",
    )

    # Mutually exclusive: either create a new catalog or connect to existing
    catalog_group = parser.add_mutually_exclusive_group(required=True)
    catalog_group.add_argument(
        "--catalog-id",
        help="Catalog ID to connect to (for existing catalogs)",
    )
    catalog_group.add_argument(
        "--create-catalog",
        metavar="PROJECT_NAME",
        help="Create a new catalog with this project name",
    )

    parser.add_argument(
        "--domain-schema",
        help="Domain schema name (auto-detected if not provided)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of images to process per batch (default: 500)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Set up schema and datasets without downloading/uploading images",
    )
    parser.add_argument(
        "--test",
        nargs="?",
        type=int,
        const=10,
        default=None,
        metavar="N",
        help="Test mode: upload only N images (default: 10 if flag used without value)",
    )
    parser.add_argument(
        "--show-urls",
        action="store_true",
        help="Show Chaise web interface URLs for datasets in the summary",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()

    # Verify Kaggle credentials
    if not args.dry_run and not verify_kaggle_credentials():
        return 1

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
