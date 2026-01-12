"""Vocabulary management tools for DerivaML MCP server."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
    from deriva_ml_mcp.connection import ConnectionManager

logger = logging.getLogger("deriva-ml-mcp")


def register_vocabulary_tools(mcp: FastMCP, conn_manager: ConnectionManager) -> None:
    """Register vocabulary management tools with the MCP server."""

    @mcp.tool()
    async def list_vocabularies() -> str:
        """List all controlled vocabulary tables in the catalog.

        Vocabularies store standardized terms (e.g., Dataset_Type, Asset_Type, Workflow_Type).

        Returns:
            JSON array of {name, schema, comment} for each vocabulary table.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            vocabs = []
            for schema in [ml.ml_schema, ml.domain_schema]:
                for table in ml.model.schemas[schema].tables.values():
                    if ml.model.is_vocabulary(table):
                        vocabs.append({
                            "name": table.name,
                            "schema": schema,
                            "comment": table.comment or "",
                        })
            return json.dumps(vocabs)
        except Exception as e:
            logger.error(f"Failed to list vocabularies: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def list_vocabulary_terms(vocabulary_name: str) -> str:
        """List all terms in a vocabulary with their descriptions and synonyms.

        Args:
            vocabulary_name: Name of the vocabulary table (e.g., "Dataset_Type", "Asset_Type").

        Returns:
            JSON array of {name, description, synonyms, rid} for each term.

        Example:
            list_vocabulary_terms("Dataset_Type") -> [{"name": "Training", ...}, ...]
        """
        try:
            ml = conn_manager.get_active_or_raise()
            terms = ml.list_vocabulary_terms(vocabulary_name)
            result = []
            for term in terms:
                result.append({
                    "name": term.name,
                    "description": term.description,
                    "synonyms": term.synonyms or [],
                    "rid": term.rid,
                })
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Failed to list terms: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def lookup_term(vocabulary_name: str, term_name: str) -> str:
        """Find a term by name or synonym in a vocabulary.

        Args:
            vocabulary_name: Name of the vocabulary table.
            term_name: Term name or any of its synonyms to search for.

        Returns:
            JSON with name, description, synonyms, rid if found.

        Example:
            lookup_term("Dataset_Type", "train") -> finds "Training" if "train" is a synonym
        """
        try:
            ml = conn_manager.get_active_or_raise()
            term = ml.lookup_term(vocabulary_name, term_name)
            return json.dumps({
                "name": term.name,
                "description": term.description,
                "synonyms": term.synonyms or [],
                "rid": term.rid,
            })
        except Exception as e:
            logger.error(f"Failed to lookup term: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def add_term(
        vocabulary_name: str,
        term_name: str,
        description: str,
        synonyms: list[str] | None = None,
    ) -> str:
        """Add a new term to a vocabulary.

        Args:
            vocabulary_name: Name of the vocabulary table (e.g., "Dataset_Type").
            term_name: Primary name for the term (must be unique).
            description: What this term means.
            synonyms: Alternative names that can also match this term.

        Returns:
            JSON with status, name, description, synonyms, rid.

        Example:
            add_term("Dataset_Type", "Validation", "Held-out data for validation", ["val", "valid"])
        """
        try:
            ml = conn_manager.get_active_or_raise()
            term = ml.add_term(
                table=vocabulary_name,
                term_name=term_name,
                description=description,
                synonyms=synonyms or [],
                exists_ok=False,
            )
            return json.dumps({
                "status": "created",
                "name": term.name,
                "description": term.description,
                "synonyms": term.synonyms or [],
                "rid": term.rid,
            })
        except Exception as e:
            if "already exists" in str(e):
                try:
                    existing = ml.lookup_term(vocabulary_name, term_name)
                    return json.dumps({
                        "status": "exists",
                        "name": existing.name,
                        "description": existing.description,
                        "rid": existing.rid,
                    })
                except Exception:
                    pass
            logger.error(f"Failed to add term: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    async def create_vocabulary(
        vocabulary_name: str,
        comment: str = "",
        schema: str | None = None,
    ) -> str:
        """Create a new vocabulary table for storing controlled terms.

        Args:
            vocabulary_name: Name for the new vocabulary table.
            comment: Description of the vocabulary's purpose.
            schema: Schema to create in (default: domain schema).

        Returns:
            JSON with status, name, schema, comment.

        Example:
            create_vocabulary("Quality_Level", "Image quality ratings")
        """
        try:
            ml = conn_manager.get_active_or_raise()
            table = ml.create_vocabulary(
                vocab_name=vocabulary_name,
                comment=comment,
                schema=schema,
            )
            return json.dumps({
                "status": "created",
                "name": table.name,
                "schema": table.schema.name,
                "comment": comment,
            })
        except Exception as e:
            logger.error(f"Failed to create vocabulary: {e}")
            return json.dumps({"status": "error", "message": str(e)})
