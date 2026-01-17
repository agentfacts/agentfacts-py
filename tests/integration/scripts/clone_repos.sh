#!/usr/bin/env bash
# Clone repositories for integration testing
# Usage: ./clone_repos.sh [framework]

set -e

REPOS_DIR="${REPOS_DIR:-"$(dirname "$0")/../repos"}"
mkdir -p "$REPOS_DIR"
export GIT_LFS_SKIP_SMUDGE=1

# Repository definitions (Bash 3-compatible)
LANGCHAIN_REPOS=(
    "langchain|https://github.com/langchain-ai/langchain.git"
    "langgraph|https://github.com/langchain-ai/langgraph.git"
    "chat-langchain|https://github.com/langchain-ai/chat-langchain.git"
    "langchain-cookbook|https://github.com/langchain-ai/langchain-cookbook.git"
    "opengpts|https://github.com/langchain-ai/opengpts.git"
)

CREWAI_REPOS=(
    "crewai|https://github.com/joaomdmoura/crewAI.git"
    "crewai-examples|https://github.com/joaomdmoura/crewAI-examples.git"
    "crewai-tools|https://github.com/joaomdmoura/crewAI-tools.git"
    "crewai-stocks|https://github.com/alejandro-ao/crewai-stocks.git"
    "trip-planner-crew|https://github.com/tonykipkemboi/trip_planner_crew.git"
)

AUTOGEN_REPOS=(
    "autogen|https://github.com/microsoft/autogen.git"
    "autogen-studio|https://github.com/microsoft/autogen-studio.git"
    "taskweaver|https://github.com/microsoft/TaskWeaver.git"
    "promptflow|https://github.com/microsoft/promptflow.git"
    "semantic-kernel|https://github.com/microsoft/semantic-kernel.git"
)

LLAMAINDEX_REPOS=(
    "llama-index|https://github.com/run-llama/llama_index.git"
    "llama-hub|https://github.com/run-llama/llama-hub.git"
    "sec-insights|https://github.com/run-llama/sec-insights.git"
    "rags|https://github.com/run-llama/rags.git"
    "create-llama|https://github.com/run-llama/create-llama.git"
)

OPENAGENTS_REPOS=(
    "openagents|https://github.com/xlang-ai/OpenAgents.git"
    "agent-eval|https://github.com/xlang-ai/agent-eval.git"
    "osworld|https://github.com/xlang-ai/OSWorld.git"
    "webarena|https://github.com/web-arena-x/webarena.git"
    "intercode|https://github.com/princeton-nlp/intercode.git"
)

HUGGINGFACE_REPOS=(
    "smolagents|https://github.com/huggingface/smolagents.git"
    "transformers|https://github.com/huggingface/transformers.git"
    "cookbook|https://github.com/huggingface/cookbook.git"
    "text-generation-inference|https://github.com/huggingface/text-generation-inference.git"
    "chat-ui|https://github.com/huggingface/chat-ui.git"
)

clone_repo() {
    local name=$1
    local url=$2
    local target_dir="$REPOS_DIR/$3/$name"
    local framework=$3
    local sparse_paths
    sparse_paths="$(sparse_paths_for_repo "$framework" "$name")"

    if [ -d "$target_dir" ]; then
        echo "  [$name] Already exists, pulling latest..."
        if [ -f "$target_dir/.git/info/sparse-checkout" ]; then
            git -C "$target_dir" sparse-checkout disable 2>/dev/null || true
        fi
        git -C "$target_dir" pull --ff-only --depth 1 2>/dev/null || true
    else
        echo "  [$name] Cloning from $url..."
        if [ -n "$sparse_paths" ]; then
            git clone --depth 1 --filter=blob:none --no-tags --sparse "$url" "$target_dir" 2>/dev/null \
                || echo "  [$name] Failed to clone"
        elif git clone --depth 1 --filter=blob:none --no-tags "$url" "$target_dir" 2>/dev/null; then
            true
        elif git clone --depth 1 --no-tags "$url" "$target_dir" 2>/dev/null; then
            true
        else
            echo "  [$name] Failed to clone"
        fi
    fi

    if [ -n "$sparse_paths" ]; then
        git -C "$target_dir" sparse-checkout init --cone 2>/dev/null || true
        git -C "$target_dir" sparse-checkout set $sparse_paths 2>/dev/null || true
        for path in $sparse_paths; do
            git -C "$target_dir" checkout HEAD -- "$path" 2>/dev/null || true
        done
    fi
}

sparse_paths_for_repo() {
    local framework=$1
    local name=$2

    case "$framework/$name" in
        langchain/langchain)
            echo "libs/core libs/langchain"
            ;;
        langchain/langgraph)
            echo "libs/langgraph"
            ;;
        crewai/crewai)
            echo "lib/crewai/src"
            ;;
        autogen/autogen)
            echo "python/packages/autogen-agentchat python/packages/autogen-core python/packages/pyautogen"
            ;;
        llamaindex/llama-index)
            echo "llama-index-core"
            ;;
        openagents/openagents)
            echo "openagents src"
            ;;
        huggingface/smolagents)
            echo "src"
            ;;
        *)
            echo ""
            ;;
    esac
}

clone_framework() {
    local framework=$1
    echo "=== Cloning $framework repositories ==="
    mkdir -p "$REPOS_DIR/$framework"

    local repos=()
    case $framework in
        langchain)
            repos=("${LANGCHAIN_REPOS[@]}")
            ;;
        crewai)
            repos=("${CREWAI_REPOS[@]}")
            ;;
        autogen)
            repos=("${AUTOGEN_REPOS[@]}")
            ;;
        llamaindex)
            repos=("${LLAMAINDEX_REPOS[@]}")
            ;;
        openagents)
            repos=("${OPENAGENTS_REPOS[@]}")
            ;;
        huggingface)
            repos=("${HUGGINGFACE_REPOS[@]}")
            ;;
        *)
            echo "Unknown framework: $framework"
            exit 1
            ;;
    esac

    for entry in "${repos[@]}"; do
        IFS="|" read -r name url <<< "$entry"
        clone_repo "$name" "$url" "$framework"
    done
}

# Main
if [ $# -eq 0 ]; then
    # Clone all frameworks
    for fw in langchain crewai autogen llamaindex openagents huggingface; do
        clone_framework "$fw"
        echo ""
    done
else
    # Clone specific framework
    clone_framework "$1"
fi

echo "=== Clone complete ==="
echo "Repositories are in: $REPOS_DIR"
