from ..agents.agent import build_search_agent, build_reader_agent,writer_chain,critic_chain


def run_research_pipeline(topic:str)->dict:
    state = {}

    #search agent working
    print("\n"+" =" *50)
    print("step 1 search agent is working...")
    print("="*50)


    search_agent = build_search_agent()
    search_result =search_agent.invoke({
        "messages":[("user",f"Find recent, reliable and detailed information about: {topic}")]
    })

    state["search_results"] = search_result["messages"][-1].content

    print("\n search result", state["search_results"])



    #Step 2 -> Reader Agent
    print("\n"+" ="*50)
    print("step 2 Reader agent is Scraping top resources")
    print("="*50)

    reader_agent = build_reader_agent()
    reader_result =reader_agent.invoke({
        "messages":[(
            "user",
            f"Based on following Search Results about '{topic}',"
            f"pick the most relevant url and scrape it deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
    })

    state["scraped_content"] = reader_result["messages"][-1].content
    print("\n scraped content", state["scraped_content"])

    #step 3 writer chain
    print("\n"+" ="*50)
    print("step 3 writer is drafting the report...")
    print("="*50)

    research_combined =(
        f"SEARCH RESULTS: \n {state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT: \n{state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic":topic,
        "research":research_combined,
    })

    print("\n Final report\n", state["report"])



    #critic report
    print("\n"+" ="*50)
    print("step 4 critic agent is working...")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })

    print("\n critic report", state["feedback"])
    return state



