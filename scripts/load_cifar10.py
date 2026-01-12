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
import shutil
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

    # Extract the zip file
    zip_files = list(download_dir.glob("*.zip"))
    if zip_files:
        logger.info("Extracting dataset...")
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(download_dir)

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
    if "Image" not in existing_tables:
        logger.info("Creating Image asset table...")
        result = await client.create_asset_table(
            asset_name="Image",
            columns=[
                {"name": "Width", "type": "int4", "comment": "Image width in pixels"},
                {"name": "Height", "type": "int4", "comment": "Image height in pixels"},
            ],
            comment="CIFAR-10 32x32 RGB images",
        )
        results["asset_table"] = result

        # Enable Image as dataset element type
        logger.info("Enabling Image as dataset element type...")
        await client.add_dataset_element_type("Image")
    else:
        logger.info("Image asset table already exists")
        results["asset_table"] = {"status": "exists", "table_name": "Image"}

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
        dataset_types=["complete"],
        version="1.0.0",
    )
    datasets["complete"] = result["rid"]
    logger.info(f"  Created Complete dataset: {result['rid']}")

    # Create Segmented dataset
    result = await client.create_dataset(
        description="CIFAR-10 dataset segmented into training and testing splits",
        dataset_types=["complete"],
        version="1.0.0",
    )
    datasets["segmented"] = result["rid"]
    logger.info(f"  Created Segmented dataset: {result['rid']}")

    # Create Training dataset
    result = await client.create_dataset(
        description="CIFAR-10 training set with 50,000 labeled images",
        dataset_types=["training"],
        version="1.0.0",
    )
    datasets["training"] = result["rid"]
    logger.info(f"  Created Training dataset: {result['rid']}")

    # Create Testing dataset
    result = await client.create_dataset(
        description="CIFAR-10 testing set",
        dataset_types=["testing"],
        version="1.0.0",
    )
    datasets["testing"] = result["rid"]
    logger.info(f"  Created Testing dataset: {result['rid']}")

    # Add Training and Testing as children of Segmented
    # Note: This adds them as members, which establishes the parent-child relationship
    await client.add_dataset_members(
        dataset_rid=datasets["segmented"],
        member_rids=[datasets["training"], datasets["testing"]],
    )
    logger.info("  Linked Training and Testing to Segmented dataset")

    return datasets


async def load_images(
    client: MCPClient,
    data_dir: Path,
    datasets: dict[str, str],
    batch_size: int = 500,
) -> dict[str, Any]:
    """Load images into the catalog using execution system.

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

        # Get working directory
        working_dir_result = await client.get_execution_working_dir()
        working_dir = Path(working_dir_result["working_dir"])
        logger.info(f"  Working directory: {working_dir}")

        # Create asset directory structure
        # Images go to: <working_dir>/Image/<Width>/<Height>/filename.png
        image_dir = working_dir / "Image" / "32" / "32"
        image_dir.mkdir(parents=True, exist_ok=True)

        # Create feature directory
        feature_dir = working_dir / "Image_Classification"
        feature_dir.mkdir(parents=True, exist_ok=True)
        feature_file = feature_dir / "Image_Classification.jsonl"

        # Load training labels
        labels = load_train_labels(data_dir)
        logger.info(f"Loaded {len(labels)} training labels")

        # Track images for dataset assignment
        train_images = []
        all_images = []

        # Process training images
        logger.info("Processing training images...")
        count = 0
        with open(feature_file, "w") as f:
            for img_path, class_name, image_id in iter_images(data_dir, "train", labels):
                if class_name is None:
                    continue

                # Create unique filename with class prefix for tracking
                new_filename = f"{class_name}_{image_id}.png"
                dest_path = image_dir / new_filename

                # Copy image
                shutil.copy(img_path, dest_path)

                # Write feature value
                feature_record = {"Filename": new_filename, "Image_Class": class_name}
                f.write(json.dumps(feature_record) + "\n")

                train_images.append(new_filename)
                all_images.append(new_filename)
                count += 1

                if count % 1000 == 0:
                    logger.info(f"  Processed {count} images...")

        logger.info(f"  Total training images: {count}")

        # Upload outputs
        logger.info("Uploading images to catalog...")
        upload_result = await client.upload_execution_outputs(clean_folder=False)
        logger.info(f"  Upload result: {upload_result}")

    finally:
        await client.stop_execution()
        logger.info("  Execution stopped")

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
    logger.info(f"Connecting to {args.hostname}, catalog {args.catalog_id}")

    async with create_mcp_client() as client:
        # Connect to catalog
        conn_result = await client.connect_catalog(
            hostname=args.hostname,
            catalog_id=str(args.catalog_id),
            domain_schema=args.domain_schema,
        )

        if conn_result.get("status") == "error":
            logger.error(f"Connection failed: {conn_result.get('message')}")
            return 1

        logger.info(f"Connected to catalog, domain schema: {conn_result.get('domain_schema')}")

        # Set up domain model
        logger.info("Setting up domain model...")
        schema_result = await setup_domain_model(client)
        logger.info(f"Domain model setup complete")

        # Create dataset hierarchy
        datasets = await create_dataset_hierarchy(client)

        if args.dry_run:
            logger.info("Dry run mode - skipping image download and upload")
            return 0

        # Download CIFAR-10 from Kaggle
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_dir = download_cifar10(temp_path)
            logger.info(f"Downloaded CIFAR-10 to: {data_dir}")

            # Load images
            load_result = await load_images(client, data_dir, datasets, args.batch_size)
            logger.info(f"Loading complete: {load_result}")

    logger.info("CIFAR-10 loading complete!")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load CIFAR-10 dataset into DerivaML catalog via MCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python load_cifar10.py --hostname ml.derivacloud.org --catalog-id 99
    python load_cifar10.py --hostname example.org --catalog-id 1 --domain-schema my_schema
    python load_cifar10.py --hostname example.org --catalog-id 1 --dry-run
        """,
    )
    parser.add_argument(
        "--hostname",
        required=True,
        help="Deriva server hostname (e.g., ml.derivacloud.org)",
    )
    parser.add_argument(
        "--catalog-id",
        required=True,
        help="Catalog ID to connect to",
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
