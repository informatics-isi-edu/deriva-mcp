"""Unit tests for MCP resources.

Tests all resources registered in resources.py:
- Server version (static)
- Configuration templates (static)
- Dynamic catalog resources (require connection)
- Parameterized resources (require connection + parameters)
- Documentation resources (mock fetch_doc)
- Storage resources (require connection)
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from deriva_ml import DerivaMLException
from deriva_ml_mcp import __version__


# =============================================================================
# Helpers
# =============================================================================


def _make_term(
    name: str = "Training",
    description: str = "Data used for training",
    synonyms: tuple[str, ...] | None = None,
    rid: str = "1-0001",
) -> MagicMock:
    """Create a mock vocabulary term object."""
    term = MagicMock()
    term.name = name
    term.description = description
    term.synonyms = synonyms if synonyms is not None else ()
    term.rid = rid
    return term


def _make_table(
    name: str = "MyTable",
    schema_name: str = "test_schema",
    comment: str = "",
    columns: list | None = None,
    is_vocabulary: bool = False,
) -> MagicMock:
    """Create a mock table object."""
    table = MagicMock()
    table.name = name
    table.schema = MagicMock()
    table.schema.name = schema_name
    table.comment = comment
    if columns is not None:
        table.columns = columns
    else:
        col = MagicMock()
        col.name = "col1"
        col.type = MagicMock()
        col.type.__str__ = lambda self: "text"
        col.type.typename = "text"
        col.nullok = True
        col.comment = ""
        table.columns = [col]
    table.is_vocabulary = is_vocabulary
    table.foreign_keys = []
    return table


def _make_dataset(
    dataset_rid: str = "DS-0001",
    description: str = "Test dataset",
    dataset_types: list | None = None,
    current_version: str = "1.0.0",
) -> MagicMock:
    ds = MagicMock()
    ds.dataset_rid = dataset_rid
    ds.description = description
    ds.dataset_types = dataset_types or ["Training"]
    ds.current_version = current_version
    return ds


def _make_workflow(
    rid: str = "WF-0001",
    name: str = "Test Workflow",
    url: str = "https://example.com/wf",
    workflow_type: str = "ML",
    description: str = "A workflow",
    checksum: str = "abc123",
    version: str = "1.0",
    is_notebook: bool = False,
) -> MagicMock:
    wf = MagicMock()
    wf.rid = rid
    wf.name = name
    wf.url = url
    wf.workflow_type = workflow_type
    wf.description = description
    wf.checksum = checksum
    wf.version = version
    wf.is_notebook = is_notebook
    return wf


def _make_asset(
    asset_rid: str = "AST-0001",
    filename: str = "model.pt",
    url: str = "https://example.com/model.pt",
    length: int = 1024,
    md5: str = "deadbeef",
    asset_types: list | None = None,
    asset_table: str = "Model",
    description: str = "A model asset",
) -> MagicMock:
    asset = MagicMock()
    asset.asset_rid = asset_rid
    asset.filename = filename
    asset.url = url
    asset.length = length
    asset.md5 = md5
    asset.asset_types = asset_types or ["Model"]
    asset.asset_table = asset_table
    asset.description = description
    asset.get_chaise_url.return_value = f"https://test.example.org/chaise/record/#1/test_schema:Model/RID={asset_rid}"
    return asset


def _make_execution_record(
    execution_rid: str = "EXE-0001",
    workflow_rid: str = "WF-0001",
    status: str = "Complete",
    description: str = "An execution",
) -> MagicMock:
    exe = MagicMock()
    exe.execution_rid = execution_rid
    exe.workflow_rid = workflow_rid
    exe.status = MagicMock()
    exe.status.value = status
    exe.description = description
    return exe


# =============================================================================
# Test Server Version
# =============================================================================


class TestServerVersion:
    """Tests for the server version resource."""

    def test_returns_version_info(self, captured_resources):
        result = captured_resources["deriva-ml://server/version"]()
        data = json.loads(result)
        assert data["name"] == "deriva-mcp"
        assert data["version"] == __version__

    def test_no_connection_still_works(self, captured_resources_disconnected):
        """Server version does not require a connection."""
        result = captured_resources_disconnected["deriva-ml://server/version"]()
        data = json.loads(result)
        assert data["name"] == "deriva-mcp"
        assert data["version"] == __version__


# =============================================================================
# Test Static Config Templates
# =============================================================================


CONFIG_TEMPLATE_URIS = [
    "deriva-ml://config/deriva-ml-template",
    "deriva-ml://config/dataset-spec-template",
    "deriva-ml://config/execution-template",
    "deriva-ml://config/model-template",
    "deriva-ml://config/experiment-template",
    "deriva-ml://config/multirun-template",
]


class TestConfigTemplates:
    """Tests for static configuration template resources."""

    @pytest.mark.parametrize("uri", CONFIG_TEMPLATE_URIS)
    def test_returns_nonempty_string(self, captured_resources, uri):
        """Each config template returns a non-empty Python code string."""
        result = captured_resources[uri]()
        assert isinstance(result, str)
        assert len(result) > 50

    @pytest.mark.parametrize("uri", CONFIG_TEMPLATE_URIS)
    def test_no_connection_still_works(self, captured_resources_disconnected, uri):
        """Config templates do not require a connection."""
        result = captured_resources_disconnected[uri]()
        assert isinstance(result, str)
        assert len(result) > 50

    def test_deriva_ml_template_contains_key_imports(self, captured_resources):
        result = captured_resources["deriva-ml://config/deriva-ml-template"]()
        assert "hydra_zen" in result
        assert "DerivaMLConfig" in result

    def test_dataset_spec_template_content(self, captured_resources):
        result = captured_resources["deriva-ml://config/dataset-spec-template"]()
        assert "DatasetSpecConfig" in result

    def test_execution_template_content(self, captured_resources):
        result = captured_resources["deriva-ml://config/execution-template"]()
        assert "ExecutionConfiguration" in result

    def test_model_template_content(self, captured_resources):
        result = captured_resources["deriva-ml://config/model-template"]()
        assert "zen_partial" in result

    def test_experiment_template_content(self, captured_resources):
        result = captured_resources["deriva-ml://config/experiment-template"]()
        assert "experiment" in result.lower()

    def test_multirun_template_content(self, captured_resources):
        result = captured_resources["deriva-ml://config/multirun-template"]()
        assert "multirun_config" in result


# =============================================================================
# Test Dynamic Catalog Resources
# =============================================================================


class TestCatalogSchema:
    """Tests for the catalog schema resource."""

    def test_success(self, captured_resources, mock_ml):
        mock_table = _make_table(name="Subject", schema_name="test_schema")
        mock_table.is_vocabulary = False

        mock_schema = MagicMock()
        mock_schema.tables = {"Subject": mock_table}

        mock_ml.model.schemas = {"test_schema": mock_schema}
        mock_ml.domain_schemas = {"test_schema"}
        mock_ml.default_schema = "test_schema"
        mock_ml.ml_schema = "deriva-ml"

        result = captured_resources["deriva-ml://catalog/schema"]()
        data = json.loads(result)

        assert data["hostname"] == "test.example.org"
        assert data["catalog_id"] == "1"
        assert "test_schema" in data["domain_schemas"]
        assert len(data["tables"]) == 1
        assert data["tables"][0]["name"] == "Subject"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://catalog/schema"]()


class TestCatalogVocabularies:
    """Tests for the catalog vocabularies resource."""

    def test_success(self, captured_resources, mock_ml):
        mock_table = _make_table(name="Dataset_Type", schema_name="deriva-ml")
        ml_schema = MagicMock()
        ml_schema.tables = {"Dataset_Type": mock_table}

        domain_schema = MagicMock()
        domain_schema.tables = {}

        mock_ml.model.schemas = {"deriva-ml": ml_schema, "test_schema": domain_schema}
        mock_ml.ml_schema = "deriva-ml"
        mock_ml.domain_schemas = {"test_schema"}
        mock_ml.model.is_vocabulary.return_value = True

        mock_term = _make_term(name="Training", description="Training data")
        mock_ml.list_vocabulary_terms.return_value = [mock_term]

        result = captured_resources["deriva-ml://catalog/vocabularies"]()
        data = json.loads(result)

        assert "Dataset_Type" in data
        assert data["Dataset_Type"][0]["name"] == "Training"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://catalog/vocabularies"]()


class TestCatalogDatasets:
    """Tests for the catalog datasets resource."""

    def test_success(self, captured_resources, mock_ml):
        ds = _make_dataset()
        mock_ml.find_datasets.return_value = [ds]

        result = captured_resources["deriva-ml://catalog/datasets"]()
        data = json.loads(result)

        assert len(data) == 1
        assert data[0]["rid"] == "DS-0001"
        assert data[0]["description"] == "Test dataset"

    def test_empty_datasets(self, captured_resources, mock_ml):
        mock_ml.find_datasets.return_value = []
        result = captured_resources["deriva-ml://catalog/datasets"]()
        data = json.loads(result)
        assert data == []

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://catalog/datasets"]()


class TestDatasetElementTypes:
    """Tests for the dataset element types resource."""

    def test_success(self, captured_resources, mock_ml):
        table = _make_table(name="Image", schema_name="test_schema", comment="Image table")
        mock_ml.list_dataset_element_types.return_value = [table]

        result = captured_resources["deriva-ml://catalog/dataset-element-types"]()
        data = json.loads(result)

        assert len(data) == 1
        assert data[0]["name"] == "Image"
        assert data[0]["schema"] == "test_schema"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://catalog/dataset-element-types"]()


class TestCatalogWorkflows:
    """Tests for the catalog workflows resource."""

    def test_success(self, captured_resources, mock_ml):
        wf = _make_workflow()
        mock_ml.find_workflows.return_value = [wf]

        result = captured_resources["deriva-ml://catalog/workflows"]()
        data = json.loads(result)

        assert len(data) == 1
        assert data[0]["rid"] == "WF-0001"
        assert data[0]["name"] == "Test Workflow"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://catalog/workflows"]()


class TestWorkflowTypes:
    """Tests for the workflow types resource."""

    def test_success(self, captured_resources, mock_ml):
        term = _make_term(name="ML Pipeline", description="ML workflow type", rid="WT-001")
        mock_ml.list_vocabulary_terms.return_value = [term]

        result = captured_resources["deriva-ml://catalog/workflow-types"]()
        data = json.loads(result)

        assert len(data) == 1
        assert data[0]["name"] == "ML Pipeline"
        assert data[0]["rid"] == "WT-001"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://catalog/workflow-types"]()


class TestCatalogFeatures:
    """Tests for the catalog features resource."""

    def test_success(self, captured_resources, mock_ml):
        term = _make_term(name="BoundingBox", description="Object bounding box")
        mock_ml.list_vocabulary_terms.return_value = [term]

        result = captured_resources["deriva-ml://catalog/features"]()
        data = json.loads(result)

        assert len(data) == 1
        assert data[0]["name"] == "BoundingBox"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://catalog/features"]()


class TestCatalogTables:
    """Tests for the catalog tables resource."""

    def test_success(self, captured_resources, mock_ml):
        table = _make_table(name="Subject", schema_name="test_schema", comment="Subject table")
        mock_schema = MagicMock()
        mock_schema.tables = {"Subject": table}
        mock_ml.model.schemas = {"test_schema": mock_schema}
        mock_ml.domain_schemas = {"test_schema"}
        mock_ml.model.is_vocabulary.return_value = False
        mock_ml.model.is_asset.return_value = False

        result = captured_resources["deriva-ml://catalog/tables"]()
        data = json.loads(result)

        assert len(data) == 1
        assert data[0]["name"] == "Subject"
        assert data[0]["is_vocabulary"] is False
        assert data[0]["is_asset"] is False

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://catalog/tables"]()


class TestDatasetTypes:
    """Tests for the dataset types resource."""

    def test_success(self, captured_resources, mock_ml):
        term = _make_term(name="Training", description="Training data", synonyms=("train",), rid="DT-001")
        mock_ml.list_vocabulary_terms.return_value = [term]

        result = captured_resources["deriva-ml://catalog/dataset-types"]()
        data = json.loads(result)

        assert len(data) == 1
        assert data[0]["name"] == "Training"
        assert data[0]["synonyms"] == ["train"]
        assert data[0]["rid"] == "DT-001"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://catalog/dataset-types"]()


class TestAssetTables:
    """Tests for the asset tables resource."""

    def test_success(self, captured_resources, mock_ml):
        table = _make_table(name="Model", schema_name="test_schema", comment="Model assets")
        mock_ml.list_asset_tables.return_value = [table]

        result = captured_resources["deriva-ml://catalog/asset-tables"]()
        data = json.loads(result)

        assert len(data) == 1
        assert data[0]["name"] == "Model"
        assert data[0]["schema"] == "test_schema"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://catalog/asset-tables"]()


class TestCatalogAssets:
    """Tests for the catalog assets summary resource."""

    def test_success(self, captured_resources, mock_ml):
        table = _make_table(name="Model", schema_name="test_schema")
        # Ensure columns have proper names for the filter
        col_rid = MagicMock()
        col_rid.name = "RID"
        col_data = MagicMock()
        col_data.name = "Filename"
        table.columns = [col_rid, col_data]

        mock_ml.model.find_assets.return_value = [table]
        asset = _make_asset()
        mock_ml.list_assets.return_value = [asset]

        result = captured_resources["deriva-ml://catalog/assets"]()
        data = json.loads(result)

        assert "Model" in data
        assert data["Model"]["count"] == 1
        # "Filename" is not in the excluded set
        assert "Filename" in data["Model"]["columns"]

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://catalog/assets"]()


class TestCatalogExecutions:
    """Tests for the catalog executions resource."""

    def test_success(self, captured_resources, mock_ml):
        mock_entity = {
            "RID": "EXE-001",
            "Workflow": "WF-001",
            "Status": "Complete",
            "Description": "Test run",
            "RCT": "2024-01-01T00:00:00",
        }
        mock_entities = MagicMock()
        mock_entities.fetch.return_value = [mock_entity]

        mock_exec_path = MagicMock()
        mock_exec_path.entities.return_value = mock_entities

        mock_schemas = MagicMock()
        mock_schemas.Execution = mock_exec_path

        mock_pb = MagicMock()
        mock_pb.schemas.__getitem__ = lambda self, key: mock_schemas

        mock_ml.pathBuilder.return_value = mock_pb
        mock_ml.ml_schema = "deriva-ml"

        result = captured_resources["deriva-ml://catalog/executions"]()
        data = json.loads(result)

        assert len(data) == 1
        assert data[0]["rid"] == "EXE-001"
        assert data[0]["status"] == "Complete"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://catalog/executions"]()


class TestCatalogExperiments:
    """Tests for the catalog experiments resource."""

    def test_success(self, captured_resources, mock_ml):
        mock_exp = MagicMock()
        mock_exp.summary.return_value = {"rid": "EXP-001", "description": "Experiment 1"}
        mock_ml.find_experiments.return_value = [mock_exp]

        result = captured_resources["deriva-ml://catalog/experiments"]()
        data = json.loads(result)

        assert data["count"] == 1
        assert data["experiments"][0]["rid"] == "EXP-001"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://catalog/experiments"]()


class TestCatalogInfo:
    """Tests for the catalog info resource."""

    def test_success(self, captured_resources, mock_ml, mock_conn_manager):
        result = captured_resources["deriva-ml://catalog/info"]()
        data = json.loads(result)

        assert data["hostname"] == "test.example.org"
        assert data["catalog_id"] == "1"
        assert data["project_name"] == "test_project"
        assert data["workflow_rid"] == "WF-TEST"
        assert data["execution_rid"] == "EXE-TEST"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://catalog/info"]()


class TestCatalogUsers:
    """Tests for the catalog users resource."""

    def test_success(self, captured_resources, mock_ml):
        mock_ml.user_list.return_value = [
            {"id": "user1", "display_name": "Alice"},
            {"id": "user2", "display_name": "Bob"},
        ]

        result = captured_resources["deriva-ml://catalog/users"]()
        data = json.loads(result)

        assert len(data) == 2
        assert data[0]["display_name"] == "Alice"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://catalog/users"]()


class TestCatalogConnections:
    """Tests for the connections resource."""

    def test_success(self, captured_resources, mock_conn_manager):
        mock_conn_manager.list_connections.return_value = [
            {"hostname": "test.example.org", "catalog_id": "1", "active": True}
        ]

        result = captured_resources["deriva-ml://catalog/connections"]()
        data = json.loads(result)

        assert len(data) == 1
        assert data[0]["active"] is True

    def test_works_without_connection(self, captured_resources_disconnected, disconnected_conn_manager):
        """Connections resource does not call get_active_or_raise, so it works even disconnected."""
        disconnected_conn_manager.list_connections.return_value = []
        result = captured_resources_disconnected["deriva-ml://catalog/connections"]()
        data = json.loads(result)
        assert data == []


# =============================================================================
# Test Dynamic Catalog Resources - Disconnected Parametrize
# =============================================================================

DYNAMIC_CATALOG_URIS = [
    "deriva-ml://catalog/schema",
    "deriva-ml://catalog/vocabularies",
    "deriva-ml://catalog/datasets",
    "deriva-ml://catalog/dataset-element-types",
    "deriva-ml://catalog/workflows",
    "deriva-ml://catalog/workflow-types",
    "deriva-ml://catalog/features",
    "deriva-ml://catalog/tables",
    "deriva-ml://catalog/dataset-types",
    "deriva-ml://catalog/asset-tables",
    "deriva-ml://catalog/assets",
    "deriva-ml://catalog/executions",
    "deriva-ml://catalog/experiments",
    "deriva-ml://catalog/info",
    "deriva-ml://catalog/users",
]


class TestDynamicCatalogDisconnected:
    """All dynamic catalog resources raise DerivaMLException when not connected."""

    @pytest.mark.parametrize("uri", DYNAMIC_CATALOG_URIS)
    def test_no_connection_raises(self, captured_resources_disconnected, uri):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected[uri]()


# =============================================================================
# Test Parameterized Resources
# =============================================================================


class TestDatasetDetails:
    """Tests for the dataset details parameterized resource."""

    def test_success(self, captured_resources, mock_ml):
        ds = _make_dataset(dataset_rid="DS-100")
        ds.list_dataset_members.return_value = {"Image": [{"RID": "IMG-1"}]}
        ds.dataset_history.return_value = []
        ds.list_dataset_children.return_value = []
        ds.list_dataset_parents.return_value = []
        mock_ml.lookup_dataset.return_value = ds

        result = captured_resources["deriva-ml://dataset/{dataset_rid}"]("DS-100")
        data = json.loads(result)

        assert data["rid"] == "DS-100"
        assert data["member_counts"]["Image"] == 1
        mock_ml.lookup_dataset.assert_called_once_with("DS-100")

    def test_with_children_and_parents(self, captured_resources, mock_ml):
        ds = _make_dataset(dataset_rid="DS-200")
        child = _make_dataset(dataset_rid="DS-201", description="Child")
        parent = _make_dataset(dataset_rid="DS-199", description="Parent")
        ds.list_dataset_members.return_value = {}
        ds.dataset_history.return_value = []
        ds.list_dataset_children.return_value = [child]
        ds.list_dataset_parents.return_value = [parent]
        mock_ml.lookup_dataset.return_value = ds

        result = captured_resources["deriva-ml://dataset/{dataset_rid}"]("DS-200")
        data = json.loads(result)

        assert len(data["children"]) == 1
        assert data["children"][0]["rid"] == "DS-201"
        assert len(data["parents"]) == 1
        assert data["parents"][0]["rid"] == "DS-199"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://dataset/{dataset_rid}"]("DS-100")


class TestDatasetMembers:
    """Tests for the dataset members parameterized resource."""

    def test_success(self, captured_resources, mock_ml):
        ds = _make_dataset(dataset_rid="DS-300")
        ds.list_dataset_members.return_value = {
            "Image": [{"RID": "IMG-1"}, {"RID": "IMG-2"}],
            "Subject": [{"RID": "SUB-1"}],
        }
        mock_ml.lookup_dataset.return_value = ds

        result = captured_resources["deriva-ml://dataset/{dataset_rid}/members"]("DS-300")
        data = json.loads(result)

        assert data["dataset_rid"] == "DS-300"
        assert data["member_counts"]["Image"] == 2
        assert data["member_counts"]["Subject"] == 1

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://dataset/{dataset_rid}/members"]("DS-300")


class TestDatasetVersions:
    """Tests for the dataset version history resource."""

    def test_success(self, captured_resources, mock_ml):
        ds = _make_dataset(dataset_rid="DS-400")
        history_entry = MagicMock()
        history_entry.version = "1.0.0"
        history_entry.description = "Initial version"
        history_entry.snapshot = "2024-01-01"
        ds.dataset_history.return_value = [history_entry]
        mock_ml.lookup_dataset.return_value = ds

        result = captured_resources["deriva-ml://dataset/{dataset_rid}/versions"]("DS-400")
        data = json.loads(result)

        assert data["dataset_rid"] == "DS-400"
        assert len(data["versions"]) == 1
        assert data["versions"][0]["version"] == "1.0.0"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://dataset/{dataset_rid}/versions"]("DS-400")


class TestTableFeatures:
    """Tests for the table features resource."""

    def test_success(self, captured_resources, mock_ml):
        feature = MagicMock()
        feature.feature_name = "BoundingBox"
        feature.target_table = MagicMock()
        feature.target_table.name = "Image"
        feature.feature_table = MagicMock()
        feature.feature_table.name = "Execution_Image_BoundingBox"
        feature.asset_columns = []
        feature.term_columns = []
        feature.value_columns = [MagicMock(name="x"), MagicMock(name="y")]
        # Ensure .name returns proper value for value_columns
        feature.value_columns[0].name = "x"
        feature.value_columns[1].name = "y"

        mock_ml.find_features.return_value = [feature]

        result = captured_resources["deriva-ml://table/{table_name}/features"]("Image")
        data = json.loads(result)

        assert len(data) == 1
        assert data[0]["name"] == "BoundingBox"
        assert data[0]["target_table"] == "Image"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://table/{table_name}/features"]("Image")


class TestFeatureDetails:
    """Tests for the feature details resource."""

    def test_success(self, captured_resources, mock_ml):
        feature = MagicMock()
        feature.feature_name = "Quality"
        feature.target_table = MagicMock()
        feature.target_table.name = "Image"
        feature.feature_table = MagicMock()
        feature.feature_table.name = "Execution_Image_Quality"
        feature.feature_table.foreign_keys = []
        feature.term_columns = []
        feature.asset_columns = []
        val_col = MagicMock()
        val_col.name = "score"
        val_col.type = MagicMock()
        val_col.type.typename = "float8"
        val_col.nullok = False
        feature.value_columns = [val_col]
        feature.optional = False

        mock_ml.lookup_feature.return_value = feature

        result = captured_resources["deriva-ml://feature/{table_name}/{feature_name}"]("Image", "Quality")
        data = json.loads(result)

        assert data["feature_name"] == "Quality"
        assert data["target_table"] == "Image"
        assert "score" in data["value_columns"]
        assert data["value_columns"]["score"]["type"] == "float8"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://feature/{table_name}/{feature_name}"]("Image", "Quality")


class TestFeatureValues:
    """Tests for the feature values resource."""

    def test_success(self, captured_resources, mock_ml):
        mock_ml.list_feature_values.return_value = [
            {"Image": "IMG-1", "score": 0.95, "Execution": "EXE-1"},
        ]

        result = captured_resources["deriva-ml://feature/{table_name}/{feature_name}/values"]("Image", "Quality")
        data = json.loads(result)

        assert len(data) == 1
        assert data[0]["Image"] == "IMG-1"
        assert data[0]["score"] == 0.95

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://feature/{table_name}/{feature_name}/values"]("Image", "Quality")


class TestVocabularyTerms:
    """Tests for the vocabulary terms resource."""

    def test_success(self, captured_resources, mock_ml):
        term = _make_term(name="Training", description="Training data", synonyms=("train",))
        mock_ml.list_vocabulary_terms.return_value = [term]

        result = captured_resources["deriva-ml://vocabulary/{vocab_name}"]("Dataset_Type")
        data = json.loads(result)

        assert len(data) == 1
        assert data[0]["name"] == "Training"
        assert data[0]["synonyms"] == ["train"]

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://vocabulary/{vocab_name}"]("Dataset_Type")


class TestVocabularyTerm:
    """Tests for the vocabulary term detail resource."""

    def test_success(self, captured_resources, mock_ml):
        term = _make_term(name="Training", description="Training data", synonyms=("train",), rid="T-001")
        mock_ml.lookup_term.return_value = term

        result = captured_resources["deriva-ml://vocabulary/{vocab_name}/{term_name}"]("Dataset_Type", "Training")
        data = json.loads(result)

        assert data["name"] == "Training"
        assert data["rid"] == "T-001"
        assert data["synonyms"] == ["train"]

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://vocabulary/{vocab_name}/{term_name}"]("Dataset_Type", "Training")


class TestTableSchema:
    """Tests for the table schema resource."""

    def test_success(self, captured_resources, mock_ml):
        col = MagicMock()
        col.name = "Name"
        col.type = MagicMock()
        col.type.typename = "text"
        col.nullok = False
        col.comment = "The name"

        table = _make_table(name="Subject", schema_name="test_schema", comment="Subject table", columns=[col])
        mock_ml.model.name_to_table.return_value = table
        mock_ml.model.is_vocabulary.return_value = False
        mock_ml.model.is_asset.return_value = False

        result = captured_resources["deriva-ml://table/{table_name}/schema"]("Subject")
        data = json.loads(result)

        assert data["name"] == "Subject"
        assert data["schema"] == "test_schema"
        assert len(data["columns"]) == 1
        assert data["columns"][0]["name"] == "Name"
        assert data["columns"][0]["type"] == "text"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://table/{table_name}/schema"]("Subject")


class TestTableAssets:
    """Tests for the table assets resource."""

    def test_success(self, captured_resources, mock_ml):
        asset = _make_asset(asset_rid="AST-100")
        mock_ml.list_assets.return_value = [asset]

        result = captured_resources["deriva-ml://table/{table_name}/assets"]("Model")
        data = json.loads(result)

        assert len(data) == 1
        assert data[0]["asset_rid"] == "AST-100"
        assert data[0]["filename"] == "model.pt"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://table/{table_name}/assets"]("Model")


class TestWorkflowDetails:
    """Tests for the workflow details resource."""

    def test_success(self, captured_resources, mock_ml):
        wf = _make_workflow(rid="WF-100", name="Test WF", is_notebook=True)
        mock_ml.lookup_workflow.return_value = wf

        result = captured_resources["deriva-ml://workflow/{workflow_rid}"]("WF-100")
        data = json.loads(result)

        assert data["rid"] == "WF-100"
        assert data["name"] == "Test WF"
        assert data["is_notebook"] is True

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://workflow/{workflow_rid}"]("WF-100")


class TestTableAnnotations:
    """Tests for the table annotations resource."""

    def test_success(self, captured_resources, mock_ml):
        mock_ml.get_table_annotations.return_value = {"display": {"name": "My Table"}}

        result = captured_resources["deriva-ml://table/{table_name}/annotations"]("Subject")
        data = json.loads(result)

        assert data["display"]["name"] == "My Table"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://table/{table_name}/annotations"]("Subject")


class TestColumnAnnotations:
    """Tests for the column annotations resource."""

    def test_success(self, captured_resources, mock_ml):
        mock_ml.get_column_annotations.return_value = {"column-display": {"*": {"markdown_pattern": "test"}}}

        result = captured_resources["deriva-ml://table/{table_name}/column/{column_name}/annotations"]("Subject", "Name")
        data = json.loads(result)

        assert "column-display" in data

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://table/{table_name}/column/{column_name}/annotations"]("Subject", "Name")


class TestTableForeignKeys:
    """Tests for the table foreign keys resource."""

    def test_success(self, captured_resources, mock_ml):
        mock_ml.list_foreign_keys.return_value = {
            "outbound": [{"from": "Subject", "to": "Species"}],
            "inbound": [],
        }

        result = captured_resources["deriva-ml://table/{table_name}/foreign-keys"]("Subject")
        data = json.loads(result)

        assert len(data["outbound"]) == 1
        assert data["outbound"][0]["to"] == "Species"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://table/{table_name}/foreign-keys"]("Subject")


class TestAssetDetails:
    """Tests for the asset details resource."""

    def test_success(self, captured_resources, mock_ml):
        asset = _make_asset(asset_rid="AST-200")
        exe_record = _make_execution_record()
        mock_ml.lookup_asset.return_value = asset
        mock_ml.list_asset_executions.return_value = [exe_record]

        result = captured_resources["deriva-ml://asset/{asset_rid}"]("AST-200")
        data = json.loads(result)

        assert data["rid"] == "AST-200"
        assert data["filename"] == "model.pt"
        assert len(data["executions"]) == 1
        assert data["executions"][0]["execution_rid"] == "EXE-0001"
        assert "chaise_url" in data

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://asset/{asset_rid}"]("AST-200")


class TestExecutionDetails:
    """Tests for the execution details resource."""

    def test_success(self, captured_resources, mock_ml):
        exe = MagicMock()
        exe.execution_rid = "EXE-500"
        exe.workflow_rid = "WF-500"
        exe.status = MagicMock()
        exe.status.value = "Complete"
        exe.configuration = MagicMock()
        exe.configuration.description = "Test execution"
        exe.list_nested_executions.return_value = [{"Nested_Execution": "EXE-501"}]
        exe.list_parent_executions.return_value = [{"Execution": "EXE-499"}]
        mock_ml.lookup_execution.return_value = exe

        result = captured_resources["deriva-ml://execution/{execution_rid}"]("EXE-500")
        data = json.loads(result)

        assert data["rid"] == "EXE-500"
        assert data["status"] == "Complete"
        assert data["nested_executions"] == ["EXE-501"]
        assert data["parent_executions"] == ["EXE-499"]

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://execution/{execution_rid}"]("EXE-500")


class TestExecutionInputs:
    """Tests for the execution inputs resource."""

    def test_success(self, captured_resources, mock_ml):
        exe = MagicMock()
        ds_input = _make_dataset(dataset_rid="DS-IN-1")
        exe.list_input_datasets.return_value = [ds_input]

        asset_input = MagicMock()
        asset_input.rid = "AST-IN-1"
        asset_input.table_name = "Model"
        asset_input.filename = "weights.pt"
        exe.list_input_assets.return_value = [asset_input]

        mock_ml.lookup_execution.return_value = exe

        result = captured_resources["deriva-ml://execution/{execution_rid}/inputs"]("EXE-600")
        data = json.loads(result)

        assert data["execution_rid"] == "EXE-600"
        assert len(data["input_datasets"]) == 1
        assert data["input_datasets"][0]["rid"] == "DS-IN-1"
        assert len(data["input_assets"]) == 1
        assert data["input_assets"][0]["rid"] == "AST-IN-1"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://execution/{execution_rid}/inputs"]("EXE-600")


class TestExperimentDetails:
    """Tests for the experiment details resource."""

    def test_success(self, captured_resources, mock_ml):
        exp = MagicMock()
        exp.summary.return_value = {"rid": "EXP-100", "description": "Experiment details"}
        mock_ml.lookup_experiment.return_value = exp

        result = captured_resources["deriva-ml://experiment/{execution_rid}"]("EXP-100")
        data = json.loads(result)

        assert data["rid"] == "EXP-100"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://experiment/{execution_rid}"]("EXP-100")


class TestChaiseUrl:
    """Tests for the chaise URL resource."""

    def test_success_table_name(self, captured_resources, mock_ml):
        mock_ml.chaise_url.return_value = "https://test.example.org/chaise/recordset/#1/test_schema:Subject"

        result = captured_resources["deriva-ml://chaise-url/{table_or_rid}"]("Subject")
        data = json.loads(result)

        assert "url" in data
        assert data["table_or_rid"] == "Subject"

    def test_success_rid_fallback(self, captured_resources, mock_ml):
        mock_ml.chaise_url.side_effect = Exception("Not a table")
        mock_result = MagicMock()
        mock_result.table = MagicMock()
        mock_result.table.schema = MagicMock()
        mock_result.table.schema.name = "test_schema"
        mock_result.table.name = "Subject"
        mock_result.rid = "ABC-123"
        mock_ml.resolve_rid.return_value = mock_result

        result = captured_resources["deriva-ml://chaise-url/{table_or_rid}"]("ABC-123")
        data = json.loads(result)

        assert "url" in data
        assert "ABC-123" in data["url"]

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://chaise-url/{table_or_rid}"]("Subject")


class TestResolveRid:
    """Tests for the RID resolution resource."""

    def test_success(self, captured_resources, mock_ml):
        mock_result = MagicMock()
        mock_result.table = MagicMock()
        mock_result.table.schema = MagicMock()
        mock_result.table.schema.name = "test_schema"
        mock_result.table.name = "Subject"
        mock_result.rid = "SUB-001"
        mock_ml.resolve_rid.return_value = mock_result

        result = captured_resources["deriva-ml://rid/{rid}"]("SUB-001")
        data = json.loads(result)

        assert data["rid"] == "SUB-001"
        assert data["table"] == "Subject"
        assert data["schema"] == "test_schema"
        assert "url" in data

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://rid/{rid}"]("SUB-001")


class TestCitationUrl:
    """Tests for the citation URL resource."""

    def test_success(self, captured_resources, mock_ml):
        mock_ml.cite.side_effect = lambda rid, current: (
            "https://test.example.org/current/SUB-001" if current
            else "https://test.example.org/permanent/SUB-001"
        )

        result = captured_resources["deriva-ml://cite/{rid}"]("SUB-001")
        data = json.loads(result)

        assert data["rid"] == "SUB-001"
        assert "permanent" in data["permanent_url"]
        assert "current" in data["current_url"]

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://cite/{rid}"]("SUB-001")


class TestRegistryResource:
    """Tests for the registry resource."""

    @patch("deriva_ml_mcp.resources.DerivaServer", create=True)
    @patch("deriva_ml_mcp.resources.get_credential", create=True)
    def test_success(self, mock_get_cred, mock_server_cls, captured_resources):
        """The registry resource uses local imports, so we patch at the resources module."""
        # We need to patch in the scope where they are imported
        mock_get_cred.return_value = {"token": "test"}

        mock_registry = MagicMock()
        mock_registry.entities.return_value.fetch.return_value = [
            {"id": "1", "name": "Test Catalog", "description": "A test", "is_persistent": True, "is_catalog": True, "deleted_on": None, "alias_target": None},
            {"id": "test-alias", "name": "My Alias", "description": "An alias", "is_catalog": False, "alias_target": "1", "deleted_on": None},
        ]

        mock_pb = MagicMock()
        mock_pb.schemas.__getitem__.return_value.tables.__getitem__.return_value = mock_registry

        mock_catalog = MagicMock()
        mock_catalog.getPathBuilder.return_value = mock_pb

        mock_server = MagicMock()
        mock_server.connect_ermrest.return_value = mock_catalog
        mock_server_cls.return_value = mock_server

        # The function imports DerivaServer and get_credential locally;
        # we need to patch at the correct location
        with patch("deriva.core.DerivaServer", mock_server_cls), \
             patch("deriva.core.get_credential", mock_get_cred):
            result = captured_resources["deriva-ml://registry/{hostname}"]("test.example.org")

        data = json.loads(result)
        assert data["hostname"] == "test.example.org"
        assert len(data["catalogs"]) == 1
        assert len(data["aliases"]) == 1


class TestAliasResource:
    """Tests for the alias resource."""

    def test_success(self, captured_resources):
        mock_server_cls = MagicMock()
        mock_alias = MagicMock()
        mock_alias.retrieve.return_value = {"id": "my-alias", "alias_target": "1", "name": "My Alias"}
        mock_server = MagicMock()
        mock_server.connect_ermrest_alias.return_value = mock_alias
        mock_server_cls.return_value = mock_server

        with patch("deriva.core.DerivaServer", mock_server_cls), \
             patch("deriva.core.get_credential", MagicMock(return_value={"token": "t"})):
            result = captured_resources["deriva-ml://alias/{hostname}/{alias_name}"]("test.example.org", "my-alias")

        data = json.loads(result)
        assert data["hostname"] == "test.example.org"
        assert data["id"] == "my-alias"


# =============================================================================
# Test Parameterized Resources - Disconnected Parametrize
# =============================================================================

PARAMETERIZED_URIS_WITH_ARGS = [
    ("deriva-ml://dataset/{dataset_rid}", ("DS-001",)),
    ("deriva-ml://dataset/{dataset_rid}/members", ("DS-001",)),
    ("deriva-ml://dataset/{dataset_rid}/versions", ("DS-001",)),
    ("deriva-ml://table/{table_name}/features", ("Image",)),
    ("deriva-ml://feature/{table_name}/{feature_name}", ("Image", "Quality")),
    ("deriva-ml://feature/{table_name}/{feature_name}/values", ("Image", "Quality")),
    ("deriva-ml://vocabulary/{vocab_name}", ("Dataset_Type",)),
    ("deriva-ml://vocabulary/{vocab_name}/{term_name}", ("Dataset_Type", "Training")),
    ("deriva-ml://table/{table_name}/schema", ("Subject",)),
    ("deriva-ml://table/{table_name}/assets", ("Model",)),
    ("deriva-ml://workflow/{workflow_rid}", ("WF-001",)),
    ("deriva-ml://table/{table_name}/annotations", ("Subject",)),
    ("deriva-ml://table/{table_name}/column/{column_name}/annotations", ("Subject", "Name")),
    ("deriva-ml://table/{table_name}/foreign-keys", ("Subject",)),
    ("deriva-ml://asset/{asset_rid}", ("AST-001",)),
    ("deriva-ml://execution/{execution_rid}", ("EXE-001",)),
    ("deriva-ml://execution/{execution_rid}/inputs", ("EXE-001",)),
    ("deriva-ml://experiment/{execution_rid}", ("EXP-001",)),
    ("deriva-ml://chaise-url/{table_or_rid}", ("Subject",)),
    ("deriva-ml://rid/{rid}", ("SUB-001",)),
    ("deriva-ml://cite/{rid}", ("SUB-001",)),
]


class TestParameterizedDisconnected:
    """All parameterized resources requiring connection raise DerivaMLException when disconnected."""

    @pytest.mark.parametrize("uri,args", PARAMETERIZED_URIS_WITH_ARGS)
    def test_no_connection_raises(self, captured_resources_disconnected, uri, args):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected[uri](*args)


# =============================================================================
# Test Documentation Resources
# =============================================================================


class TestAnnotationContextsDoc:
    """Tests for the static annotation contexts documentation resource."""

    def test_returns_valid_json(self, captured_resources):
        result = captured_resources["deriva-ml://docs/annotation-contexts"]()
        data = json.loads(result)

        assert "visible_columns_contexts" in data
        assert "visible_foreign_keys_contexts" in data
        assert "table_display_contexts" in data
        assert "column_display_contexts" in data

    def test_contains_key_contexts(self, captured_resources):
        result = captured_resources["deriva-ml://docs/annotation-contexts"]()
        data = json.loads(result)

        contexts = data["visible_columns_contexts"]["contexts"]
        assert "*" in contexts
        assert "compact" in contexts
        assert "detailed" in contexts
        assert "entry" in contexts

    def test_no_connection_still_works(self, captured_resources_disconnected):
        """Annotation contexts is static, no connection needed."""
        result = captured_resources_disconnected["deriva-ml://docs/annotation-contexts"]()
        data = json.loads(result)
        assert "visible_columns_contexts" in data


DOC_URIS_AND_FETCH_ARGS = [
    ("deriva-ml://docs/overview", "deriva-ml", "docs/user-guide/overview.md"),
    ("deriva-ml://docs/datasets", "deriva-ml", "docs/user-guide/datasets.md"),
    ("deriva-ml://docs/features", "deriva-ml", "docs/user-guide/features.md"),
    ("deriva-ml://docs/execution-configuration", "deriva-ml", "docs/user-guide/execution-configuration.md"),
    ("deriva-ml://docs/hydra-zen", "deriva-ml", "docs/user-guide/hydra-zen-configuration.md"),
    ("deriva-ml://docs/file-assets", "deriva-ml", "docs/user-guide/file-assets.md"),
    ("deriva-ml://docs/notebooks", "deriva-ml", "docs/user-guide/notebooks.md"),
    ("deriva-ml://docs/annotations", "deriva-ml", "docs/user-guide/annotations.md"),
    ("deriva-ml://docs/identifiers", "deriva-ml", "docs/user-guide/identifiers.md"),
    ("deriva-ml://docs/install", "deriva-ml", "docs/user-guide/install.md"),
    ("deriva-ml://docs/ermrest/data-api", "ermrest", "docs/api-doc/data/rest.md"),
    ("deriva-ml://docs/ermrest/naming", "ermrest", "docs/api-doc/data/naming.md"),
    ("deriva-ml://docs/ermrest/catalog", "ermrest", "docs/api-doc/rest-catalog.md"),
    ("deriva-ml://docs/chaise/config", "chaise", "docs/user-docs/chaise-config.md"),
    ("deriva-ml://docs/chaise/query-parameters", "chaise", "docs/user-docs/query-parameters.md"),
    ("deriva-ml://docs/deriva-py/install", "deriva-py", "docs/install.md"),
    ("deriva-ml://docs/deriva-py/tutorial", "deriva-py", "docs/project-tutorial.md"),
]


class TestDocResources:
    """Tests for documentation resources that use fetch_doc."""

    @pytest.mark.parametrize("uri,repo,path", DOC_URIS_AND_FETCH_ARGS)
    def test_calls_fetch_doc(self, captured_resources, uri, repo, path):
        """Each doc resource calls fetch_doc with the right repo and path."""
        with patch("deriva_ml_mcp.resources.fetch_doc", return_value="# Mock Doc Content") as mock_fetch:
            result = captured_resources[uri]()
            mock_fetch.assert_called_once_with(repo, path)
            assert result == "# Mock Doc Content"

    @pytest.mark.parametrize("uri,repo,path", DOC_URIS_AND_FETCH_ARGS)
    def test_no_connection_still_works(self, captured_resources_disconnected, uri, repo, path):
        """Doc resources do not require a connection."""
        with patch("deriva_ml_mcp.resources.fetch_doc", return_value="# Doc") as mock_fetch:
            result = captured_resources_disconnected[uri]()
            mock_fetch.assert_called_once_with(repo, path)
            assert result == "# Doc"


# =============================================================================
# Test Storage Resources
# =============================================================================


class TestStorageSummary:
    """Tests for the storage summary resource."""

    def test_success(self, captured_resources, mock_ml):
        mock_ml.get_storage_summary.return_value = {
            "total_size": "1.5 GB",
            "cache_size": "500 MB",
            "execution_size": "1.0 GB",
        }

        result = captured_resources["deriva-ml://storage/summary"]()
        data = json.loads(result)

        assert data["total_size"] == "1.5 GB"
        assert data["cache_size"] == "500 MB"

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://storage/summary"]()


class TestCacheStats:
    """Tests for the cache stats resource."""

    def test_success(self, captured_resources, mock_ml):
        mock_ml.get_cache_size.return_value = {
            "size_bytes": 500000,
            "size_human": "500 KB",
            "file_count": 10,
        }

        result = captured_resources["deriva-ml://storage/cache"]()
        data = json.loads(result)

        assert data["size_bytes"] == 500000
        assert data["file_count"] == 10

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://storage/cache"]()


class TestExecutionDirs:
    """Tests for the execution directories resource."""

    def test_success(self, captured_resources, mock_ml):
        mock_ml.list_execution_dirs.return_value = [
            {"path": "/tmp/exe-001", "size": 1024, "modified": datetime(2024, 1, 1, 12, 0, 0)},
        ]

        result = captured_resources["deriva-ml://storage/execution-dirs"]()
        data = json.loads(result)

        assert data["count"] == 1
        assert data["execution_dirs"][0]["path"] == "/tmp/exe-001"
        # modified should be converted to ISO format
        assert "2024-01-01" in data["execution_dirs"][0]["modified"]

    def test_no_connection(self, captured_resources_disconnected):
        with pytest.raises(DerivaMLException):
            captured_resources_disconnected["deriva-ml://storage/execution-dirs"]()


# =============================================================================
# Test All Resources Are Registered
# =============================================================================


class TestResourceRegistration:
    """Verify that all expected resources are registered."""

    EXPECTED_URIS = [
        # Server info
        "deriva-ml://server/version",
        # Config templates
        "deriva-ml://config/deriva-ml-template",
        "deriva-ml://config/dataset-spec-template",
        "deriva-ml://config/execution-template",
        "deriva-ml://config/model-template",
        "deriva-ml://config/experiment-template",
        "deriva-ml://config/multirun-template",
        # Dynamic catalog resources
        "deriva-ml://catalog/schema",
        "deriva-ml://catalog/vocabularies",
        "deriva-ml://catalog/datasets",
        "deriva-ml://catalog/dataset-element-types",
        "deriva-ml://catalog/workflows",
        "deriva-ml://catalog/workflow-types",
        "deriva-ml://catalog/features",
        "deriva-ml://catalog/tables",
        "deriva-ml://catalog/dataset-types",
        "deriva-ml://catalog/asset-tables",
        "deriva-ml://catalog/assets",
        "deriva-ml://catalog/executions",
        "deriva-ml://catalog/experiments",
        "deriva-ml://catalog/info",
        "deriva-ml://catalog/users",
        "deriva-ml://catalog/connections",
        # Parameterized resources
        "deriva-ml://dataset/{dataset_rid}",
        "deriva-ml://dataset/{dataset_rid}/members",
        "deriva-ml://dataset/{dataset_rid}/versions",
        "deriva-ml://table/{table_name}/features",
        "deriva-ml://feature/{table_name}/{feature_name}",
        "deriva-ml://feature/{table_name}/{feature_name}/values",
        "deriva-ml://vocabulary/{vocab_name}",
        "deriva-ml://vocabulary/{vocab_name}/{term_name}",
        "deriva-ml://table/{table_name}/schema",
        "deriva-ml://table/{table_name}/assets",
        "deriva-ml://workflow/{workflow_rid}",
        "deriva-ml://table/{table_name}/annotations",
        "deriva-ml://table/{table_name}/column/{column_name}/annotations",
        "deriva-ml://table/{table_name}/foreign-keys",
        "deriva-ml://asset/{asset_rid}",
        "deriva-ml://execution/{execution_rid}",
        "deriva-ml://execution/{execution_rid}/inputs",
        "deriva-ml://experiment/{execution_rid}",
        "deriva-ml://chaise-url/{table_or_rid}",
        "deriva-ml://rid/{rid}",
        "deriva-ml://cite/{rid}",
        "deriva-ml://registry/{hostname}",
        "deriva-ml://alias/{hostname}/{alias_name}",
        # Doc resources
        "deriva-ml://docs/annotation-contexts",
        "deriva-ml://docs/overview",
        "deriva-ml://docs/datasets",
        "deriva-ml://docs/features",
        "deriva-ml://docs/execution-configuration",
        "deriva-ml://docs/hydra-zen",
        "deriva-ml://docs/file-assets",
        "deriva-ml://docs/notebooks",
        "deriva-ml://docs/annotations",
        "deriva-ml://docs/identifiers",
        "deriva-ml://docs/install",
        "deriva-ml://docs/ermrest/data-api",
        "deriva-ml://docs/ermrest/naming",
        "deriva-ml://docs/ermrest/catalog",
        "deriva-ml://docs/chaise/config",
        "deriva-ml://docs/chaise/query-parameters",
        "deriva-ml://docs/deriva-py/install",
        "deriva-ml://docs/deriva-py/tutorial",
        # Storage resources
        "deriva-ml://storage/summary",
        "deriva-ml://storage/cache",
        "deriva-ml://storage/execution-dirs",
    ]

    @pytest.mark.parametrize("uri", EXPECTED_URIS)
    def test_resource_is_registered(self, captured_resources, uri):
        """Every expected resource URI is present in captured_resources."""
        assert uri in captured_resources, f"Resource {uri} not registered"

    def test_no_unexpected_resources(self, captured_resources):
        """No unexpected resources are registered beyond the expected set."""
        registered = set(captured_resources.keys())
        expected = set(self.EXPECTED_URIS)
        unexpected = registered - expected
        assert not unexpected, f"Unexpected resources registered: {unexpected}"
