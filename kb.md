# Hardware Dashboard Focus

- Hardware identified as the main focus going forward
- Dashboard needed for control and officer visibility
- Framed as a narrative push around American models, where confidence is higher than on the hardware side itself

# Sensor Data Collection Plan

- Data to be sourced from two places:
  - Hackathon team sensors
  
  - Own sensors
- Proposed sequence: collect data first, then build on top of it
- One participant considering shifting effort toward sensor work to support the other before splitting off

# 5G and AI Benchmarking

- 5G / AI benchmark work mentioned as a parallel workstream
- Effort allocation still being weighed between benchmark work and sensor collaboration

# Ambient Notes

- Brief discussion of AI search tools: Gemini used to find a nearby breakfast spot
- DeepSeek Flash 4.1 mentioned: loses to Claude, only ~10% faster than UltraFast
- RSI referenced as a topic of shared interest; someone relevant may be worth reaching out to




----- 


# Project Context

- Hackathon project: soldier wellness monitoring using provided sensors
- Focus track: prolonged field care, injury prevention during training
- Sensor inventory available:
  - ECG/EKG (electrical heart signal)
  - PPG (photoplethysmography)
  - EDA/GSR (electrodermal activity, galvanic skin response)
  - Skin temperature
  - CO2 (air-based)
  - IMU/gyroscope (likely not needed)

# Recommended Sensor Configuration

- Best signals for the use case: EDA/GSR + skin temperature
  - EDA/GSR measures sweat gland activity via skin conductance
  - Skin temperature tracks heat load directly
- CO2 sensor: air-based, so cannot directly monitor blood O2 conversion
- ECG: useful but form factor is a challenge (chest/neck placement impractical under 60-200 lb tactical load)
- Interstitial fluid sensors would be ideal but not available
  - Interstitial fluid reflects cortisol, glucose, stress, and heart rate changes in real time

# Soldier Perspective: Real Gaps in the Field

- Training environment is the primary use case, not active missions
- Biggest training risk: heat casualties
  - Soldiers operate in 105-120°F for extended hours
  - Body shuts down when it cannot cool itself
- Key need: a pre-warning system that monitors temperature and sweat, then alerts when rest is needed
- Formation/mission context: sensors would be too intrusive; training-side application is the right fit

# LLM and AI Limitations for Biomedical Data

- Strong pushback on using LLMs to analyze or model soldier physiology
- Too many unaccounted variables:
  - Metabolic rate, genetic disorders, diet, tobacco/alcohol use, mental distress, hormonal changes
- Neuroplasticity analogy: neuron connections change daily, so any mapping is immediately outdated
  - A live, continuously updating AI translator would be needed, which is not feasible
- LLMs described as “a better dictionary”: useful for finding connections and keywords, not modeling biological reality
- Recommended approach: collect data, then manually analyze with domain experts and doctors

# Recommended Direction and Next Steps

- Sensor system should serve as a pre-injury warning system
- Output data (sweat levels, temperature, ECG) can be shown to doctors as reference points
  - E.g., sweat produced before a heat collapse event as a baseline marker
- Goal: monitor signals continuously, flag risk thresholds, support training plan improvement

# Next Steps

- **Email G with follow-up questions on sensor design and use cases**

---

Chat with meeting transcript: https://notes.granola.ai/t/5cc3179a-fc62-46b3-80c4-98e7c091a994


----- 
# Hackathon Overview

- Biodefense-themed hackathon with extensive sensor kit provided
- 5 tracks: Frontier Bio for the Battlefield, Human Performance and Prolonged Field Care, Biological Data for Defense, Biosecurity and AI-Guided Countermeasures, Distributed Biomanufacturing
- $10K first prize per track (5 tracks total)
- Judges are biotech/defense-oriented, not pure science academics
- Key constraint: not bio experts, so ideas must be buildable with sensors on hand

# Three Ideas Evaluated

- Idea 1: Fine-tune open American models to outperform Chinese adversarial AI
  - Train in RL environments; defense framing around AI biosecurity
  - Assessed as weakest of the three
- Idea 2: Protein bonding RL environment
  - Goes beyond AlphaFold (folding only) to optimize proteins that bind to a target substructure
  - Applications: bioweapon countermeasures, vaccine design, targeted drug delivery
  - Viable only with protein design expertise; keep as alternate
- Idea 3: Distributed phenotyping via minimal sensors
  - Use many sensors to find which single sensor best predicts all others
  - Build a lightweight, single-sensor wearable from that result
  - Assessed as strongest fit for the hackathon

# Pivot: Merge Ideas 2 and 3 (Core Concept)

- Wearable monitors soldier vitals and activity (accelerometer, gyroscope, heart rate, ECG)
- Detects when a soldier is down via fluctuating vitals and loss of movement
- Triggers targeted protein/stimulant delivery to stabilize the casualty
  - Inspired by Orexin receptor research cited in the hackathon prompt
  - Delivery gated behind human approval: buzzer + button confirm, or voice command (“I’m hit”)
- Central operator dashboard shows:
  - Live location of all warfighters
  - Who is active, injured, or critical
  - Triage priority for extraction
- GPS: use phone/laptop API for demo; AirTag or similar as fallback
  - GPS-denied mode discussed but deprioritized for now

# Product and Demo Strategy

- Build a physical sensor prototype (chest strap or arm strap)
- Demo: hook up all sensors, show single-sensor reconstruction of full signal suite
  - Display combined vs. estimated charts to prove accuracy
- Protein delivery mechanism shown conceptually, not physically injected
- Avoid pure research/benchmark framing: judges want a buildable product, not a model
- Normalize vitals relative to individual baseline to generalize across soldiers

# Sensor Kit Available

- ECG, PPG, heart rate, IMU, skin temperature, force-sensitive resistors
- 60 GHz radar, thermal camera, CO2/VOC/particulate sensors, 11-channel spectral sensor
- Polar H10 chest strap, Velostat sheet, USB night vision webcam
- Breadboards, I2C wiring, soldering stations on site
- Personal sensors (accelerometer, gyroscope) can be brought from dorm

# Next Steps

- **Submit project proposal with 3 ideas**
- **Scope and validate the merged Idea 2+3 concept**
- **Bring personal sensor kit (accelerometer, gyroscope) to hackathon**
- **Research judges' backgrounds to sharpen pitch angle**

---

Chat with meeting transcript: https://notes.granola.ai/t/9ae0ff66-23de-4e06-a10a-e630c680e8a0

The first San Diego Defense × Bio hackathon. 24 hours straight. $10K to the winning team.

Five Tracks:

- Frontier Bio for the Battlefield and the Homefront

- Biological Data for Defense

- Human Performance & Prolonged Field Care

- Distributed Biomanufacturing

- Biosecurity & AI-guided Countermeasures

​Day 1: Briefings from stakeholders on Defense and national capability gaps begin at 11:00am. Teams begin building at noon with 24 hours to complete the build.

Day 2: Demos begin at noon on Sunday (in-person). $10K first prize.

​The last hackathon like this in SoCal launched multiple billion-dollar defense companies.

​Hosted by American Bioenhancement Corp. Sponsored by 1517 Fund, Pilgrim Labs, Biosphere, Satomic, Adaptyv, and Permanence Bio.

​Approval required. Apply below. After approval you'll be onboarded to the Hackathon Slack for team formation.
Blasts
Toby Myers
Today, 9:53 AM

​WIFI:

​We have 2 networks in the building that will be active for the full 24 hours. 317 is active now and we’ll bring 318 online once the building begins.

​1.

​WiFiTnet-317

​Password: Wifit@7856

​2.

​WiFiTnet-318

​Password: Wifit@5687

​-Toby
Toby Myers
Yesterday, 2:18 PM

​COMPETITORS:

​Less than 24 hours until the building begins. If you have not yet joined the slack, join here:

​https://join.slack.com/t/2026sddefense-dcs5018/shared_invite/zt-4ar9avjo2-Bz9MS2n5kNxjzW1x4roreg?referrer=luma&utm_source=luma

​We’ve compiled a list of provided hardware, which we’ll share in the slack shortly. Hardware is first come-first served.

​Worth repeating: if you plan to form your team onsite, make sure to arrive early so you have time to register. All teams and builds must register before the event.

​Bring whatever you need to build for 24 hours straight, including small snacks if needed (some will be provided). There is extra space in the facility upstairs that can serve as a rest area.
Toby Myers
Yesterday, 2:10 PM
Hackathon Announcement

​We’ve added several speakers to the list, including former DARPA leaders and ex-special operations personnel.

​Rumor has it there will be an ex-Army Ranger manning a grill in the parking lot, and plenty of electrolytes and caffeine/nootropics onsite.

​We expect the event to be pretty packed, especially at registration & kickoff, so plan accordingly. Don’t forget, ID is strictly required for entry!

​COMPETITORS:

​If you plan to form your team onsite, make sure to arrive early so you have time to register. All teams and builds must register.

​Bring whatever you need to build for 24 hours straight, including small snacks if needed (some will be provided). There is extra space in the facility upstairs that can serve as a rest area.
Toby Myers
Oct 14, 8:00 AM
Full event schedule, food, and logistics

​Hi everyone, we're super excited to kick things off on Saturday.

Some more information for the day of:

Food at the facility will be minimal and no food of any kind is allowed in the lab, but we will provide plenty of caffeinated beverages and small items.

The Craft Taco is across right the street and has some of San Diego's best gourmet tacos. They will be providing a discount to all attendees (via digital code you can show at the counter). Plenty of good food is available within walking distance.

Parking will be available in the lots onsite subject to tenant priority (it's a weekend, so more spots should be available). Please be as efficient as possible with carpooling.

SATURDAY:
11578 Sorrento Valley Road, Suite 25
Registration is at 9:30am, and briefings begin at 11:00am sharp.
PHOTO ID IS REQUIRED FOR ENTRY.

Speakers: Spencer Macdonald (ex-Pentagon + ABE), Alan Tomusiak (Permanence Bio), Warren Fogler (ex-Navy EOD), Arye Lipman (BioSphere), Jake Adler (Pilgrim Labs), Corey Howe (Adaptyv Bio), and Edward You (ex-FBI Biosecurity agent).

​Hackathon kick-off at 12:00pm - the 24-hour build clock begins.

SUNDAY:

Building continues until 12:00pm.

Demos begin immediately at noon (onsite).

Judges and final winner selection will be at 2:00pm!
Toby Myers
Oct 13, 8:30 AM
Important info for the hackathon

​We’re excited to welcome you to the DefenseBio hackathon.

​Address is 11578 Sorrento Valley Rd, Suite 25, doors open at 9:30am for early check in. Event schedule has been posted on the Luma page.

Photo ID is required for check-in. Everyone entering the building must be approved on Luma, no exceptions. Please arrive with enough time to check in before the briefings begin.

We've finalized our list of judges:
Arye Lipman of Biosphere, Jake Adler of Pilgrim Labs, Moufeed Kaddoura of 1517 fund, Edward You (former FBI biosecurity agent), and Joseph Alzagatiti of American Bioenhancement Corp.

Competitors: if you haven't joined the slack yet, the link is here: https://join.slack.com/t/2026sddefense-dcs5018/shared_invite/zt-4ar9avjo2-Bz9MS2n5kNxjzW1x4roreg?referrer=luma

Feel free to reach out to the event organizers via slack if you have questions on team formation, builds, or anything else.
Toby Myers
Sep 23, 10:52 PM
Approved Competitors: Join the Slack to get started!

​We’ll be using Slack for event updates, team coordination, logistics, and announcements.

​If you are competing, please join the slack as soon as you can:

​https://join.slack.com/t/2026sddefense-dcs5018/shared_invite/zt-4ar9avjo2-Bz9MS2n5kNxjzW1x4roreg

​-ABE Team
Spencer Macdonald
Sep 22, 4:08 PM
Welcome and Onboarding for San Diego Defense X Bio Hackathon 2026

​Afternoon, all.

​Thank you for registering for the SD Defense X Bio Hackathon 2026, sponsored by 1517, Pilgrim Labs, Satomic, and Adaptyv. 

​Things to put on your radar: 

​1. Slack invites will go live shortly to the email you registered with

​2. Prompts for projects to build on during the hackathon are in the Slack channel announcements

​3. When you form your team, please submit the Google Form (ONE PER TEAM) by 10/10/2026, also found in the Slack

​Any questions or concerns, paste to the general-questions in Slack. 

​See you all soon! 

​-ABE Team
Location





