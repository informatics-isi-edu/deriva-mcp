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

        Returns a list of vocabulary tables that can be used to
        store standardized terms.

        Returns:
            JSON array of vocabulary table names.
        """
        try:
            ml = conn_manager.get_active_or_raise()
            # Get all tables and filter for vocabularies
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
        """List all terms in a vocabulary table.

        Returns all terms, descriptions, and synonyms from the
        specified controlled vocabulary.

        Args:
            vocabulary_name: Name of the vocabulary table.

        Returns:
            JSON array of term objects with name, description, and synonyms.
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
        """Look up a specific term in a vocabulary.

        Finds a term by its name or any of its synonyms.

        Args:
            vocabulary_name: Name of the vocabulary table.
            term_name: Name or synonym of the term to find.

        Returns:
            JSON object with term details or error if not found.
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

        Creates a standardized term with description and optional synonyms.

        Args:
            vocabulary_name: Name of the vocabulary table.
            term_name: Primary name for the term (must be unique in vocabulary).
            description: Description of the term's meaning.
            synonyms: Optional alternative names for the term.

        Returns:
            JSON object with created term details.
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
            # Check if term already exists
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
        """Create a new controlled vocabulary table.

        Creates a table for storing standardized terms with their
        descriptions and synonyms.

        Args:
            vocabulary_name: Name for the new vocabulary table.
            comment: Description of the vocabulary's purpose.
            schema: Schema to create the table in (defaults to domain schema).

        Returns:
            JSON object with created vocabulary details.
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
