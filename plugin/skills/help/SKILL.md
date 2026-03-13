---
name: help
description: "Use this skill when the user asks general questions about DerivaML, Deriva, deriva-mcp, or what they can do with these tools — including 'what is DerivaML', 'how do I use Deriva', 'what can you help me with', 'how does this work', or 'where do I start'. Also trigger for broad orientation questions about catalogs, datasets, experiments, hydra-zen configuration, ML workflows, or the MCP server when the user seems to be asking 'how do I approach this' rather than requesting a specific action. ALWAYS prefer this skill for general 'what/how/why' questions about the DerivaML ecosystem before routing to more specific skills."
---

# DerivaML Capabilities Guide

When the user asks what's possible or needs orientation, present the following guide. Tailor your response to their context — if they mention a specific area, focus on that section. If they're brand new, give the full overview.

## What I Can Help You With

### Set Up Your Environment
- Set up a new DerivaML project from a template
- Install Jupyter kernels and configure notebook dependencies
- Authenticate with Deriva/Globus
- **Check if your DerivaML ecosystem is up to date** — checks all three components (deriva-ml Python package, deriva-mcp skills plugin, and deriva-mcp MCP server) against upstream releases and offers to update outdated ones. Run `/deriva:check-versions` or just ask *"check versions"*
- Configure linting, docstrings, and coding standards

**Just ask:** *"help me set up my environment"*, *"am I up to date?"*, or *"check deriva versions"*

### Define Your Catalog Structure
- Create tables for your domain data (images, subjects, samples, etc.)
- Create asset tables for storing files (images, model weights, CSVs)
- Add columns, foreign keys, and constraints
- Set up controlled vocabularies with terms and synonyms
- Customize how tables appear in the Chaise web UI

**Just ask:** *"create a table for patient images"* or *"set up a vocabulary for diagnosis types"*

### Explore Your Data
- Query and filter catalog tables
- Look up records by RID
- Count records, sample data, browse vocabularies
- Understand your catalog schema

**Just ask:** *"show me the first 20 images where Diagnosis is Normal"* or *"what tables are in my catalog?"*

### Organize Data for ML
- Create datasets and add members from catalog tables
- Split datasets into training/testing/validation partitions
- Create features for labeling and annotation (classification, ground truth, confidence scores)
- Manage dataset versions for reproducibility
- Download and prepare data for ML frameworks (denormalize, BDBag, restructure for PyTorch)
- Track asset provenance — find which execution created a file

**Just ask:** *"create a labeled dataset and split it 80/20"* or *"denormalize my dataset into a DataFrame"*

### Run Experiments
- Run ML experiments with full provenance tracking
- Configure experiment presets and hyperparameter sweeps
- Do dry runs to test configuration before committing
- Run Jupyter notebooks with execution tracking
- Create new model functions and wire them into the project
- Write and validate Hydra-Zen configuration files

**Just ask:** *"run the cifar10_quick experiment"* or *"create a new model for image classification"*

### Troubleshoot Problems
- Debug execution failures (authentication, timeouts, missing files)
- Fix stuck executions
- Diagnose missing data in dataset exports
- Resolve version mismatches

**Just ask:** *"my execution is stuck in Running"* or *"my dataset bag is missing images"*

### Write Scripts for Catalog Operations
- Generate Python scripts for batch data loading, ETL, and feature population
- Scripts include provenance tracking and dry-run support
- Committed scripts ensure reproducibility

**Just ask:** *"write a script to load annotations from a CSV"*

## Tips

- **You don't need to know command names** — just describe what you want in plain language
- **I'll guide you through the steps** — each capability includes best practices and common pitfalls
- **Start with a connected catalog** — most operations need a catalog connection first. Just say *"connect to my catalog"*
- **Use dry runs** when experimenting — add "dry run" to any request to preview without making changes
