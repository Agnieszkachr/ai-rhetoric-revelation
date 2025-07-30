# src/config.py

"""
================================================================================
Configuration File for the Revelation AI Rhetoric Analysis Project
================================================================================

This file contains all the key settings to run the analytical notebooks.
By modifying the variables in this file, you can control which Bible text,
which literary structure, and which AI model is used for the analysis.

Instructions:
1.  To switch models, uncomment the desired provider block and make sure all
    other provider blocks are commented out.
2.  Ensure you have a .env file in the project's root directory with the
    necessary API keys (e.g., GOOGLE_API_KEY="your_key_here").
"""

# ------------------------------------------------------------------------------
# CORE ANALYTICAL LEXICON
# ------------------------------------------------------------------------------
# This list defines the bespoke eight-vector rhetorical framework for the study.
# WARNING: Do not modify this list if you want to reproduce the results from
# the original paper. Changing these values will fundamentally alter the analysis.
ANALYTICAL_CONCEPTS_LIST = [
    "Worship & Praise",
    "Judicial Wrath & Punitive Action",
    "Lament, Persecution & Endurance",
    "Victory, Consolation & New-Creation Hope",
    "Cosmic Warfare & Deception",
    "Prophetic Exhortation & Warning",
    "Theophanic Awe & Terror",
    "Other/Neutral Content"
]

# ------------------------------------------------------------------------------
# ANALYSIS TARGET SELECTION
# ------------------------------------------------------------------------------
# These settings determine which text and structure to analyze.
# - STRUCTURE_NAME: 'osborne' or 'aune' (for the robustness check).
# - BIBLE_NAME: 'greek' (SBLGNT) or 'nrsv' (for the translation comparison).

STRUCTURE_NAME = "osborne"
BIBLE_NAME = "greek"

# ------------------------------------------------------------------------------
# AI PROVIDER AND MODEL SELECTION
# ------------------------------------------------------------------------------
# This is the main section for configuring the AI model.
# To activate a provider, remove the '#' from the beginning of its lines.
# To deactivate a provider, add a '#' to the beginning of its lines.
#
# >>> IMPORTANT: ONLY ONE PROVIDER CAN BE ACTIVE AT A TIME. <<<
#

# --- Google Gemini (Primary model for the paper) ------------------------------
# Used for the main analysis in the published study.
PROVIDER          = "gemini"
AI_MODEL          = "gemini-2.5-pro"  # 
API_SECRET_NAME   = "GOOGLE_API_KEY" # Looks for this key in your .env file

# --- GroqCloud (Llama 3) ------------------------------------------------------
# Provides very fast inference with Llama models. Used for cross-model validation.
#PROVIDER          = "groq"
#AI_MODEL          = "llama3-70b-8192"
#API_SECRET_NAME   = "GROQ_API_KEY" # Looks for this key in your .env file

# --- Fireworks AI (Llama 3) ---------------------------------------------------
# Another provider for open-source models like Llama 3.
# PROVIDER          = "fireworks"
# AI_MODEL          = "accounts/fireworks/models/llama-v3-70b-instruct"
# API_SECRET_NAME   = "FIREWORKS_API_KEY" # Looks for this key in your .env file

# --- Cloudflare Workers AI ----------------------------------------------------
# Note: This provider requires an additional account ID.
#PROVIDER          = "cloudflare"
#AI_MODEL          = "@cf/meta/llama-3-8b-instruct"
#API_SECRET_NAME   = "CLOUDFLARE_API_KEY" # Looks for this key in your .env file
#CLOUDFLARE_ACCOUNT_ID = "YOUR_CLOUDFLARE_ACCOUNT_ID" # Add your Cloudflare ID here

# ------------------------------------------------------------------------------
# FILE SYSTEM PATHS
# ------------------------------------------------------------------------------
# These paths define the project's directory structure.
# They are relative to the location of the scripts in the /src directory.
# It is not recommended to change these unless you have reorganized the project.
PATH_RESULTS = '../data/results'
PATH_PROCESSED = '../data/processed'
PATH_INPUT = '../data/input'
PATH_FIGURES = '../figures'
