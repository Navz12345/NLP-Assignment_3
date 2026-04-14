from __future__ import annotations

from pathlib import Path

from sequential_tuning.utils.io import write_json


TOPICS = [
    "sports",
    "ai_trends",
    "news",
    "crypto",
    "oil_prices",
    "food",
    "environment",
    "technology",
    "entertainment",
    "domestic_robots",
    "mental_health_ai",
    "autonomous_commuting",
    "healthcare_ops",
    "education_tools",
    "travel_planning",
]

TOPIC_POOL = TOPICS * 10



def _make_extraction(idx: int, topic: str, split: str) -> dict:
    templates = {
        "sports": (
            "I copied this from a sports roundup. Extract the teams, player names, locations, dates, and numeric stats into valid JSON.",
            "On 2026-03-12, the Dallas Wings beat the Chicago Sky in Austin, 88 to 81. Arike Ogunbowale scored 29 points and hit 5 three-pointers.",
        ),
        "ai_trends": (
            "Read this AI industry note and extract organizations, model names, dates, and numeric claims into JSON.",
            "On 2026-02-18, OpenAI and Anthropic were both mentioned in a briefing about model efficiency. The note claimed a 22 percent reduction in inference cost for model Atlas-4.",
        ),
        "news": (
            "Turn this short news blurb into JSON by extracting people, places, organizations, dates, and quoted amounts.",
            "Mayor Elena Ruiz spoke in San Antonio on 2026-01-09 and announced a $14 million flood-repair plan with the Texas Water Board.",
        ),
        "crypto": (
            "Extract assets, prices, dates, exchanges, and percentage changes from the market note into valid JSON.",
            "On 2026-03-04, Bitcoin traded near 84250 dollars on Coinbase while Ether moved to 4620 dollars, up 3.8 percent for the day.",
        ),
        "oil_prices": (
            "Extract product names, benchmark names, dates, prices, and percentage movements from this energy update.",
            "Brent crude closed at 84.10 dollars on 2026-02-11 while WTI settled at 80.95 dollars, down 1.3 percent after inventory data.",
        ),
        "food": (
            "Read the food feature and extract dish names, restaurants, neighborhoods, and prices into JSON.",
            "At Nopal House in Austin, the mole enchiladas were listed at 18 dollars and the tres leches cake at 9 dollars during the 2026 spring menu preview.",
        ),
        "environment": (
            "Extract locations, dates, environmental measurements, and organizations from this climate note.",
            "In Phoenix on 2026-04-02, the Air Quality Council reported PM2.5 at 41 micrograms per cubic meter after a dust event.",
        ),
        "technology": (
            "Extract companies, devices, release dates, and specs from the product summary into JSON.",
            "NovaTech introduced the PixelForge X2 laptop on 2026-02-27 with 32 GB RAM, a 1 TB SSD, and a starting price of 1499 dollars.",
        ),
        "entertainment": (
            "Pull out film titles, actor names, platforms, dates, and review scores from the note into JSON.",
            "The film Midnight Harbor starring Naomi Scott arrived on StreamBox on 2026-03-20 and received an 88 critic score.",
        ),
        "domestic_robots": (
            "Extract robot model names, home tasks, dates, and performance numbers into JSON.",
            "HomeMate R3 completed 47 kitchen cleanups and 19 laundry-sort tasks in a March 2026 trial in Seattle.",
        ),
        "mental_health_ai": (
            "Extract app names, age groups, dates, and reported metrics from this mental-health product note.",
            "CalmPath AI launched a teen support pilot on 2026-01-14 and reported a 17 percent increase in weekly check-ins after four weeks.",
        ),
        "autonomous_commuting": (
            "Extract company names, cities, dates, route lengths, and safety numbers into valid JSON.",
            "MetroDrive tested its autonomous shuttle in Denver on 2026-03-08 across a 12 mile route and reported zero collisions over 640 trips.",
        ),
        "healthcare_ops": (
            "Extract hospitals, dates, departments, and operational metrics from this update.",
            "Riverside Medical Center reduced MRI wait time from 19 days to 11 days in February 2026 after adding weekend scheduling.",
        ),
        "education_tools": (
            "Extract school names, product names, dates, student counts, and improvement metrics from the note.",
            "North Ridge High adopted LearnLoop Tutor on 2026-01-22 for 430 students and reported a 9 percent increase in quiz completion.",
        ),
        "travel_planning": (
            "Extract destinations, dates, hotels, prices, and trip durations from this travel note.",
            "A four-night Kyoto trip from 2026-05-10 to 2026-05-14 included a stay at Cedar Lantern Hotel for 182 dollars per night.",
        ),
    }
    instruction, input_text = templates[topic]
    return {
        "split": split,
        "task_type": "json_extraction",
        "instruction": instruction,
        "input": f"{input_text} Reference ID EX{idx:03d}.",
        "schema": {"entities": "array", "dates": "array", "numbers": "array"},
        "reference_output": '{"entities": [], "dates": [], "numbers": []}',
    }


def _make_schema_generation(idx: int, topic: str, split: str) -> dict:
    prompts = {
        "sports": "Create a valid JSON object for a match summary card about tonight's basketball game.",
        "ai_trends": "Create a valid JSON object for a short AI trend briefing card.",
        "news": "Create a valid JSON object for a newsroom alert item.",
        "crypto": "Create a valid JSON object for a crypto watchlist alert.",
        "oil_prices": "Create a valid JSON object for an energy desk market snapshot.",
        "food": "Create a valid JSON object for a restaurant recommendation card.",
        "environment": "Create a valid JSON object for an environmental incident summary.",
        "technology": "Create a valid JSON object for a product launch summary.",
        "entertainment": "Create a valid JSON object for a streaming release card.",
        "domestic_robots": "Create a valid JSON object for a home robot task report.",
        "mental_health_ai": "Create a valid JSON object for a mental-health assistant session summary.",
        "autonomous_commuting": "Create a valid JSON object for an autonomous transit trip record.",
        "healthcare_ops": "Create a valid JSON object for a hospital workflow summary.",
        "education_tools": "Create a valid JSON object for a classroom software rollout.",
        "travel_planning": "Create a valid JSON object for a travel itinerary card.",
    }
    return {
        "split": split,
        "task_type": "schema_generation",
        "instruction": prompts[topic] + " Follow the schema exactly and return only JSON.",
        "input": f"Topic: {topic}. Make it realistic but concise. Include a title, priority, and whether follow-up is needed. Prompt ID SG{idx:03d}.",
        "schema": {"title": "string", "priority": "string", "follow_up_needed": "boolean"},
        "reference_output": '{"title": "Example", "priority": "medium", "follow_up_needed": false}',
    }


def _make_classification(idx: int, topic: str, split: str) -> dict:
    messages = {
        "sports": "The team looked flat in the second half, but the defense improved late and the road win still counts.",
        "ai_trends": "This startup is not launching a new model yet, but it is clearly positioning itself for enterprise AI infrastructure.",
        "news": "This update reads like a public safety notice rather than a political statement.",
        "crypto": "I am excited about the rally, but I also think this move looks overheated and risky in the short term.",
        "oil_prices": "Supply headlines are pushing prices up again, even though demand signals still look mixed.",
        "food": "The place is charming and the desserts are memorable, but the service fell apart when it got busy.",
        "environment": "This report is mainly about air quality deterioration and public health concerns after the dust storm.",
        "technology": "The product is polished, but the battery life claims feel more like marketing than reality.",
        "entertainment": "The movie is visually impressive, though the script loses momentum halfway through.",
        "domestic_robots": "The robot handled sweeping well, but it struggled with clutter and voice commands.",
        "mental_health_ai": "The app sounds supportive, but I would want stronger privacy controls before recommending it.",
        "autonomous_commuting": "The shuttle system feels promising for downtown commuting, though the route coverage is still limited.",
        "healthcare_ops": "This message is clearly about scheduling delays and patient throughput rather than equipment failure.",
        "education_tools": "Teachers liked the dashboard, but students said the reminders felt repetitive and easy to ignore.",
        "travel_planning": "This request is mostly about budget-conscious trip planning with flexible dates.",
    }
    return {
        "split": split,
        "task_type": "json_classification",
        "instruction": "Classify the text into one of these labels: positive, mixed, negative, informational. Return JSON only.",
        "input": f"{messages[topic]} Case CL{idx:03d}.",
        "schema": {"label": "string", "confidence": "number"},
        "reference_output": '{"label": "mixed", "confidence": 0.88}',
    }


def _make_repair(idx: int, topic: str, split: str) -> dict:
    broken = {
        "sports": '{"topic": "sports", "team": "Dallas Wings", "status": "final"',
        "ai_trends": '{"topic": "ai_trends", "model": "Atlas-4", "cost_change": -22,',
        "news": '{"topic": "news", "city": "San Antonio", "severity": "medium"',
        "crypto": '{"topic": "crypto", "asset": "BTC", "price": 84250,',
        "oil_prices": '{"topic": "oil_prices", "benchmark": "Brent", "price": 84.10',
        "food": '{"topic": "food", "dish": "mole enchiladas", "price": 18,',
        "environment": '{"topic": "environment", "metric": "PM2.5", "value": 41',
        "technology": '{"topic": "technology", "device": "PixelForge X2", "ram_gb": 32,',
        "entertainment": '{"topic": "entertainment", "title": "Midnight Harbor", "score": 88',
        "domestic_robots": '{"topic": "domestic_robots", "robot": "HomeMate R3", "tasks": 66,',
        "mental_health_ai": '{"topic": "mental_health_ai", "app": "CalmPath AI", "pilot": true',
        "autonomous_commuting": '{"topic": "autonomous_commuting", "route_miles": 12, "trips": 640',
        "healthcare_ops": '{"topic": "healthcare_ops", "department": "MRI", "wait_days": 11',
        "education_tools": '{"topic": "education_tools", "students": 430, "completion_gain": 9',
        "travel_planning": '{"topic": "travel_planning", "destination": "Kyoto", "nights": 4',
    }
    return {
        "split": split,
        "task_type": "json_repair",
        "instruction": "Repair the malformed JSON and return only the corrected JSON object.",
        "input": broken[topic] + f" /* RP{idx:03d} */",
        "schema": {"topic": "string"},
        "reference_output": '{"topic": "' + topic + '"}',
    }


def _make_tool_call(idx: int, topic: str, split: str) -> dict:
    requests = {
        "sports": "Check tonight's score and top performers for the Dallas Wings game.",
        "ai_trends": "Find the latest headlines about efficient multimodal AI models from the last 7 days.",
        "news": "Pull the latest local flood-preparedness headlines for San Antonio.",
        "crypto": "Get the latest Bitcoin and Ether prices and 24 hour percentage changes.",
        "oil_prices": "Show the latest Brent and WTI prices with daily movement.",
        "food": "Find three famous barbecue places in Austin that are open now.",
        "environment": "Get this week's air quality trend for Phoenix.",
        "technology": "Find recent reviews of the PixelForge X2 laptop.",
        "entertainment": "Show me tonight's top streaming movies in the thriller category.",
        "domestic_robots": "Find the latest reviews of home cleaning robots under 700 dollars.",
        "mental_health_ai": "Search for articles about AI-driven mental health support for college students.",
        "autonomous_commuting": "Find recent news about autonomous shuttles in Denver.",
        "healthcare_ops": "Look up case studies on reducing MRI scheduling delays.",
        "education_tools": "Find recent reports on AI tutoring tools in high schools.",
        "travel_planning": "Search for four-night hotel options in Kyoto in mid May.",
    }
    return {
        "split": split,
        "task_type": "tool_call_generation",
        "instruction": "Convert the request into a tool-call JSON object with a tool name and named arguments.",
        "input": requests[topic] + f" Request ID TC{idx:03d}.",
        "schema": {"tool_name": "string", "arguments": "object"},
        "reference_output": '{"tool_name": "search", "arguments": {}}',
    }


def build_human_seed_dataset(output_path: str, train_per_task: int = 40, eval_per_task: int = 20) -> dict:
    rows: list[dict] = []
    builders = [
        _make_extraction,
        _make_schema_generation,
        _make_classification,
        _make_repair,
        _make_tool_call,
    ]
    idx = 1
    for builder in builders:
        for topic in TOPIC_POOL[:train_per_task]:
            rows.append(builder(idx, topic, "train"))
            idx += 1
        for topic in TOPIC_POOL[train_per_task : train_per_task + eval_per_task]:
            rows.append(builder(idx, topic, "eval"))
            idx += 1
    write_json(rows, output_path)
    return {
        "output_path": str(Path(output_path)),
        "total_count": len(rows),
        "train_count": sum(1 for row in rows if row["split"] == "train"),
        "eval_count": sum(1 for row in rows if row["split"] == "eval"),
    }

