"""Integration tests for the CIFAR-10 loader script.

These tests require a local Deriva server running on localhost.
They create a test catalog, run the CIFAR-10 loader, and verify
that the expected schema, vocabulary, and datasets are created.

To run these tests:
    pytest tests/test_load_cifar10.py -v -m integration

Requirements:
    - Local Deriva server running on localhost
    - Network access (for MCP server communication)
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from deriva_ml import DerivaML

# Import the loader components
from scripts.load_cifar10 import (
    CIFAR10_CLASSES,
    MCPClient,
    create_dataset_hierarchy,
    create_mcp_client,
    load_images,
    setup_domain_model,
    setup_workflow_type,
)


# Skip all tests if not running integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def localhost_server() -> str:
    """Return the localhost server hostname."""
    return os.environ.get("DERIVA_TEST_HOST", "localhost")


@pytest.fixture(scope="module")
def event_loop():
    """Create an event loop for the test module."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def project_name() -> str:
    """Generate a unique project name for the test catalog."""
    return f"cifar10_test_{os.getpid()}"


@pytest.fixture(scope="module")
async def mcp_client_with_catalog(localhost_server: str, project_name: str):
    """Create an MCP client and a test catalog.

    Creates a new DerivaML catalog using the MCP create_catalog tool,
    yields the client for testing, then deletes the catalog afterwards.
    """
    async with create_mcp_client() as client:
        # Create a new test catalog using MCP tool
        result = await client.call_tool(
            "create_catalog",
            {"hostname": localhost_server, "project_name": project_name},
        )

        if result.get("status") == "error":
            raise RuntimeError(f"Failed to create catalog: {result.get('message')}")

        catalog_id = result["catalog_id"]

        try:
            yield client, catalog_id, localhost_server
        finally:
            # Clean up: delete the catalog using MCP tool
            try:
                await client.call_tool(
                    "delete_catalog",
                    {"hostname": localhost_server, "catalog_id": catalog_id},
                )
            except Exception as e:
                print(f"Warning: Failed to delete test catalog: {e}")


@pytest.fixture(scope="module")
async def mcp_client(mcp_client_with_catalog) -> MCPClient:
    """Extract just the MCP client from the fixture."""
    client, _, _ = mcp_client_with_catalog
    return client


@pytest.fixture(scope="module")
def catalog_id(mcp_client_with_catalog) -> str:
    """Extract the catalog ID from the fixture."""
    _, catalog_id, _ = mcp_client_with_catalog
    return catalog_id


@pytest.fixture(scope="module")
def deriva_ml(localhost_server: str, catalog_id: str) -> DerivaML:
    """Create a DerivaML instance for the test catalog."""
    return DerivaML(
        hostname=localhost_server,
        catalog_id=catalog_id,
        check_auth=False,
    )


@pytest.fixture(scope="module")
def sample_cifar_data() -> Generator[Path, None, None]:
    """Create a small sample of CIFAR-10-like data for testing.

    Instead of downloading the full dataset from Kaggle, create
    a small set of test images to verify the loader works correctly.
    """
    import csv
    from PIL import Image

    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / "cifar10"
        train_dir = data_dir / "train"
        train_dir.mkdir(parents=True)

        # Create a small set of sample images (2 per class = 20 total)
        labels = {}
        image_id = 1
        for class_name, _, _ in CIFAR10_CLASSES:
            for i in range(2):
                # Create a 32x32 image with a unique color per class
                class_idx = [c[0] for c in CIFAR10_CLASSES].index(class_name)
                color = ((class_idx * 25) % 256, (i * 50) % 256, 128)
                img = Image.new("RGB", (32, 32), color)

                img_path = train_dir / f"{image_id}.png"
                img.save(img_path)
                labels[str(image_id)] = class_name
                image_id += 1

        # Write labels file
        labels_file = data_dir / "trainLabels.csv"
        with open(labels_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "label"])
            writer.writeheader()
            for img_id, label in labels.items():
                writer.writerow({"id": img_id, "label": label})

        yield data_dir


class TestCIFAR10LoaderIntegration:
    """Integration tests for the CIFAR-10 loader."""

    @pytest.mark.asyncio
    async def test_setup_domain_model_creates_vocabulary(
        self, mcp_client: MCPClient
    ):
        """Test that setup_domain_model creates the Image_Class vocabulary."""
        result = await setup_domain_model(mcp_client)

        # Should have created vocabulary
        assert "vocabulary" in result

        # Verify vocabulary exists via MCP
        vocabs = await mcp_client.list_vocabularies()
        vocab_names = {v["name"] for v in vocabs}
        assert "Image_Class" in vocab_names

    @pytest.mark.asyncio
    async def test_setup_domain_model_creates_all_class_terms(
        self, mcp_client: MCPClient
    ):
        """Test that all 10 CIFAR-10 class terms are created."""
        await setup_domain_model(mcp_client)

        # Verify all terms exist
        terms = await mcp_client.list_vocabulary_terms("Image_Class")
        term_names = {t["name"] for t in terms}

        expected_classes = {c[0] for c in CIFAR10_CLASSES}
        assert term_names == expected_classes, (
            f"Missing terms: {expected_classes - term_names}, "
            f"Extra terms: {term_names - expected_classes}"
        )

    @pytest.mark.asyncio
    async def test_setup_domain_model_creates_image_asset_table(
        self, mcp_client: MCPClient
    ):
        """Test that setup_domain_model creates the Image asset table."""
        await setup_domain_model(mcp_client)

        # Verify Image table exists
        tables = await mcp_client.list_tables()
        table_names = {t["name"] for t in tables}
        assert "Image" in table_names

    @pytest.mark.asyncio
    async def test_setup_domain_model_is_idempotent(
        self, mcp_client: MCPClient
    ):
        """Test that running setup_domain_model twice doesn't cause errors."""
        # Run twice - should not raise
        result1 = await setup_domain_model(mcp_client)
        result2 = await setup_domain_model(mcp_client)

        # Second run should indicate things already exist
        assert result2["vocabulary"].get("status") == "exists" or "Image_Class" in str(result2)

    @pytest.mark.asyncio
    async def test_create_dataset_hierarchy(
        self, mcp_client: MCPClient
    ):
        """Test that create_dataset_hierarchy creates all expected datasets."""
        datasets = await create_dataset_hierarchy(mcp_client)

        # Should have all four datasets
        assert "complete" in datasets
        assert "segmented" in datasets
        assert "training" in datasets
        assert "testing" in datasets

        # All should have valid RIDs
        for name, rid in datasets.items():
            assert rid is not None, f"Dataset {name} has no RID"
            assert len(rid) > 0, f"Dataset {name} has empty RID"

    @pytest.mark.asyncio
    async def test_load_images_with_sample_data(
        self,
        mcp_client: MCPClient,
        sample_cifar_data: Path,
    ):
        """Test loading sample images into the catalog."""
        # First set up the domain model
        await setup_domain_model(mcp_client)

        # Create dataset hierarchy
        datasets = await create_dataset_hierarchy(mcp_client)

        # Load sample images
        result = await load_images(
            client=mcp_client,
            data_dir=sample_cifar_data,
            datasets=datasets,
            batch_size=10,
        )

        # Should have loaded 20 images (2 per class x 10 classes)
        assert result["total_images"] == 20
        assert result["training_images"] == 20

    @pytest.mark.asyncio
    async def test_loaded_images_have_image_asset_type(
        self,
        mcp_client: MCPClient,
        sample_cifar_data: Path,
    ):
        """Test that uploaded images have the 'Image' asset type."""
        # Set up and load
        await setup_domain_model(mcp_client)
        datasets = await create_dataset_hierarchy(mcp_client)
        await load_images(mcp_client, sample_cifar_data, datasets, batch_size=10)

        # Get all images
        assets = await mcp_client.list_assets("Image")

        # Should have images
        assert len(assets) > 0, "No images were uploaded"

        # Each image should have Asset_Type containing "Image"
        for asset in assets:
            asset_types = asset.get("Asset_Type", [])
            assert "Image" in asset_types or any("Image" in str(t) for t in asset_types), (
                f"Asset {asset.get('RID')} missing 'Image' asset type, has: {asset_types}"
            )

    @pytest.mark.asyncio
    async def test_images_assigned_to_complete_dataset(
        self,
        mcp_client: MCPClient,
        deriva_ml: DerivaML,
        sample_cifar_data: Path,
    ):
        """Test that all images are assigned to the Complete dataset."""
        # Set up and load
        await setup_domain_model(mcp_client)
        datasets = await create_dataset_hierarchy(mcp_client)
        await load_images(mcp_client, sample_cifar_data, datasets, batch_size=10)

        # Get complete dataset members
        complete_dataset = deriva_ml.lookup_dataset(datasets["complete"])
        members = complete_dataset.list_dataset_members()

        # Should have Image members
        assert "Image" in members, "No Image members in Complete dataset"
        assert len(members["Image"]) == 20, (
            f"Expected 20 images in Complete dataset, got {len(members['Image'])}"
        )

    @pytest.mark.asyncio
    async def test_images_assigned_to_training_dataset(
        self,
        mcp_client: MCPClient,
        deriva_ml: DerivaML,
        sample_cifar_data: Path,
    ):
        """Test that training images are assigned to the Training dataset."""
        # Set up and load
        await setup_domain_model(mcp_client)
        datasets = await create_dataset_hierarchy(mcp_client)
        await load_images(mcp_client, sample_cifar_data, datasets, batch_size=10)

        # Get training dataset members
        training_dataset = deriva_ml.lookup_dataset(datasets["training"])
        members = training_dataset.list_dataset_members()

        # Should have Image members
        assert "Image" in members, "No Image members in Training dataset"
        assert len(members["Image"]) == 20, (
            f"Expected 20 images in Training dataset, got {len(members['Image'])}"
        )

    @pytest.mark.asyncio
    async def test_segmented_dataset_has_children(
        self,
        mcp_client: MCPClient,
        deriva_ml: DerivaML,
        sample_cifar_data: Path,
    ):
        """Test that Segmented dataset has Training and Testing as children."""
        # Set up and load
        await setup_domain_model(mcp_client)
        datasets = await create_dataset_hierarchy(mcp_client)

        # Get segmented dataset
        segmented_dataset = deriva_ml.lookup_dataset(datasets["segmented"])
        children = segmented_dataset.list_dataset_children()

        # Should have two children
        assert len(children) == 2, f"Expected 2 children, got {len(children)}"

        # Children should include training and testing RIDs
        child_rids = {c["RID"] for c in children} if isinstance(children[0], dict) else set(children)
        assert datasets["training"] in child_rids or any(
            datasets["training"] in str(c) for c in children
        ), "Training dataset not a child of Segmented"
        assert datasets["testing"] in child_rids or any(
            datasets["testing"] in str(c) for c in children
        ), "Testing dataset not a child of Segmented"


class TestCIFAR10ClassVocabulary:
    """Tests specifically for the Image_Class vocabulary."""

    @pytest.mark.asyncio
    async def test_vocabulary_has_correct_term_count(
        self, mcp_client: MCPClient
    ):
        """Test that Image_Class vocabulary has exactly 10 terms."""
        await setup_domain_model(mcp_client)

        terms = await mcp_client.list_vocabulary_terms("Image_Class")
        assert len(terms) == 10, f"Expected 10 terms, got {len(terms)}"

    @pytest.mark.asyncio
    async def test_each_class_has_description(
        self, mcp_client: MCPClient
    ):
        """Test that each class term has a description."""
        await setup_domain_model(mcp_client)

        terms = await mcp_client.list_vocabulary_terms("Image_Class")

        for term in terms:
            assert term.get("description"), (
                f"Term {term['name']} has no description"
            )

    @pytest.mark.asyncio
    async def test_class_names_match_cifar10(
        self, mcp_client: MCPClient
    ):
        """Test that class names exactly match CIFAR-10 classes."""
        await setup_domain_model(mcp_client)

        terms = await mcp_client.list_vocabulary_terms("Image_Class")
        term_names = sorted(t["name"] for t in terms)

        expected = sorted(["airplane", "automobile", "bird", "cat", "deer",
                          "dog", "frog", "horse", "ship", "truck"])

        assert term_names == expected, (
            f"Term names don't match. Got: {term_names}, Expected: {expected}"
        )


class TestDatasetTypes:
    """Tests for dataset type assignments."""

    @pytest.mark.asyncio
    async def test_complete_dataset_type(
        self,
        mcp_client: MCPClient,
        deriva_ml: DerivaML,
    ):
        """Test that Complete dataset has 'complete' type."""
        await setup_domain_model(mcp_client)
        datasets = await create_dataset_hierarchy(mcp_client)

        complete = deriva_ml.lookup_dataset(datasets["complete"])
        assert "complete" in complete.dataset_types, (
            f"Complete dataset missing 'complete' type, has: {complete.dataset_types}"
        )

    @pytest.mark.asyncio
    async def test_training_dataset_type(
        self,
        mcp_client: MCPClient,
        deriva_ml: DerivaML,
    ):
        """Test that Training dataset has 'training' type."""
        await setup_domain_model(mcp_client)
        datasets = await create_dataset_hierarchy(mcp_client)

        training = deriva_ml.lookup_dataset(datasets["training"])
        assert "training" in training.dataset_types, (
            f"Training dataset missing 'training' type, has: {training.dataset_types}"
        )

    @pytest.mark.asyncio
    async def test_testing_dataset_type(
        self,
        mcp_client: MCPClient,
        deriva_ml: DerivaML,
    ):
        """Test that Testing dataset has 'testing' type."""
        await setup_domain_model(mcp_client)
        datasets = await create_dataset_hierarchy(mcp_client)

        testing = deriva_ml.lookup_dataset(datasets["testing"])
        assert "testing" in testing.dataset_types, (
            f"Testing dataset missing 'testing' type, has: {testing.dataset_types}"
        )
