#!/usr/bin/env bash
# Quick start script for coverage bot
# 
# This script can use either:
# 1. GitHub CLI (gh) - default method
# 2. GitHub API with PAT token - if GH_PAT_COVERAGE_BOT env var is set
#
# Usage:
#   bash scripts/quick_bot_start.sh                    # Use gh CLI
#   GH_PAT_COVERAGE_BOT=token bash scripts/quick_bot_start.sh  # Use API
#
set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🤖 Coverage Bot Quick Start${NC}"
echo "================================"

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo -e "${RED}❌ GitHub CLI (gh) not installed${NC}"
    echo "Install: https://cli.github.com/"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo -e "${RED}❌ Not authenticated with GitHub${NC}"
    echo "Run: gh auth login"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo -e "📍 Current branch: ${YELLOW}$CURRENT_BRANCH${NC}"

# Check if PR needs merging
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo -e "\n${YELLOW}📋 Checking PR status...${NC}"
    PR_NUMBER=$(gh pr view --json number -q .number 2>/dev/null || echo "")
    
    if [ -n "$PR_NUMBER" ]; then
        echo -e "Found PR #$PR_NUMBER"
        echo -e "${YELLOW}Merge this PR first? (y/n)${NC}"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            gh pr merge "$PR_NUMBER" --squash
            git checkout main
            git pull
        fi
    fi
fi

# Show current coverage
echo -e "\n${YELLOW}📊 Current Coverage Status${NC}"
if [ -f coverage.xml ]; then
    COVERAGE=$(python3 -c "import xml.etree.ElementTree as ET; print(f\"{float(ET.parse('coverage.xml').getroot().get('line-rate', 0)) * 100:.2f}%\")" 2>/dev/null || echo "Unknown")
    echo -e "Coverage: ${GREEN}$COVERAGE${NC}"
else
    echo -e "Coverage: ${RED}No data${NC}"
fi

# Check secrets
echo -e "\n${YELLOW}🔐 Checking GitHub Secrets...${NC}"
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)

# List of required secrets
REQUIRED_SECRETS=(
    "GH_PAT_COVERAGE_BOT"
    "COV_TARGET"
)

OPTIONAL_SECRETS=(
    "COV_GAIN_MIN"
    "GDRIVE_FILE_ID"
    "GDRIVE_SHA256"
    "SLACK_COV_WEBHOOK"
)

echo -e "Repository: ${GREEN}$REPO${NC}"
echo -e "\nRequired secrets:"
for secret in "${REQUIRED_SECRETS[@]}"; do
    echo -e "  - $secret"
done

echo -e "\nOptional secrets:"
for secret in "${OPTIONAL_SECRETS[@]}"; do
    echo -e "  - $secret"
done

echo -e "\n${YELLOW}⚠️  Make sure these are set in:${NC}"
echo -e "https://github.com/$REPO/settings/secrets/actions"

# Check if PAT token is available
if [ -n "${GH_PAT_COVERAGE_BOT:-}" ]; then
    echo -e "\n${GREEN}✅ Using PAT token for API calls${NC}"
    USE_PAT=true
else
    echo -e "\n${YELLOW}⚠️  PAT token not found in environment${NC}"
    echo -e "Using gh CLI authentication instead"
    USE_PAT=false
fi

# Function to trigger workflow via API
trigger_workflow_api() {
    local owner=$(echo "$REPO" | cut -d'/' -f1)
    local repo_name=$(echo "$REPO" | cut -d'/' -f2)
    
    echo -e "\n${YELLOW}🔧 Triggering workflow via API...${NC}"
    
    local response=$(curl -s -X POST \
        -H "Accept: application/vnd.github+json" \
        -H "Authorization: Bearer $GH_PAT_COVERAGE_BOT" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        -H "User-Agent: coverage-bot" \
        "https://api.github.com/repos/$owner/$repo_name/actions/workflows/coverage-bot.yml/dispatches" \
        -d '{"ref":"main"}' \
        -w "\n%{http_code}")
    
    local http_code=$(echo "$response" | tail -n1)
    
    if [ "$http_code" = "204" ]; then
        echo -e "${GREEN}✅ Workflow triggered successfully!${NC}"
        return 0
    else
        echo -e "${RED}❌ Failed to trigger workflow (HTTP $http_code)${NC}"
        echo -e "Response: $(echo "$response" | head -n-1)"
        return 1
    fi
}

# Prompt to run workflow
echo -e "\n${YELLOW}🚀 Ready to start coverage bot?${NC}"
echo -e "This will:"
echo -e "  1. Check current coverage"
echo -e "  2. Generate new tests if < target"
echo -e "  3. Create PR if improvement found"
echo -e "\n${GREEN}Run workflow? (y/n)${NC}"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo -e "\n${YELLOW}🤖 Starting coverage bot...${NC}"
    
    if [ "$USE_PAT" = true ] && [ -n "${GH_PAT_COVERAGE_BOT:-}" ]; then
        # Use API with PAT token
        if trigger_workflow_api; then
            echo -e "\n${GREEN}✅ Workflow started via API!${NC}"
        else
            echo -e "${YELLOW}Falling back to gh CLI...${NC}"
            gh workflow run coverage-bot.yml --ref main
        fi
    else
        # Use gh CLI
        gh workflow run coverage-bot.yml --ref main
        echo -e "\n${GREEN}✅ Workflow started!${NC}"
    fi
    
    echo -e "View progress at: https://github.com/$REPO/actions"
    
    # Wait and show status
    echo -e "\n${YELLOW}⏳ Waiting for workflow to start...${NC}"
    sleep 5
    
    # Show latest runs
    echo -e "\n${YELLOW}📋 Latest workflow runs:${NC}"
    gh run list --workflow=coverage-bot.yml --limit 3
else
    echo -e "\n${YELLOW}Skipped workflow run${NC}"
    echo -e "You can run it manually later with:"
    echo -e "  ${GREEN}gh workflow run coverage-bot.yml --ref main${NC}"
    echo -e "Or with PAT token:"
    echo -e "  ${GREEN}export GH_PAT_COVERAGE_BOT=your_token${NC}"
    echo -e "  ${GREEN}bash $0${NC}"
fi

echo -e "\n${GREEN}🎯 Next steps:${NC}"
echo -e "1. Wait ~5-10 minutes for bot to complete"
echo -e "2. Check for new PR at: https://github.com/$REPO/pulls"
echo -e "3. Review and merge the PR"
echo -e "4. Bot will run again tomorrow at 3:30 AM UTC"

echo -e "\n${YELLOW}💡 Pro tip:${NC}"
echo -e "Watch live logs with: ${GREEN}gh run watch${NC}"