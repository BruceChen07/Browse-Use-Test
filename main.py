from browser_use import Agent, Browser, ChatBrowserUse
import asyncio


async def example():
    browser = Browser(
        # use_cloud=True,  # Uncomment to use a stealth browser on Browser Use Cloud
    )

    llm = ChatBrowserUse()

    agent = Agent(
        task="Download service now icon from bing, the picture format should be png",
        llm=llm,
        browser=browser,
    )

    history = await agent.run()
    return history

if __name__ == "__main__":
    history = asyncio.run(example())
