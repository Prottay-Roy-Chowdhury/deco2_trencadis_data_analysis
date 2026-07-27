from gh_agent.config import (
    LOCAL_GH_AGENT_HOST,
)

from gh_agent.gh_agent import (
    GrasshopperAgent,
)


DESIGN_GH_AGENT_PORT = 6005


def main():
    agent = GrasshopperAgent(
        host=LOCAL_GH_AGENT_HOST,
        port=DESIGN_GH_AGENT_PORT,
        project_action_target="design_pc",
    )

    agent.start()


if __name__ == "__main__":
    main()