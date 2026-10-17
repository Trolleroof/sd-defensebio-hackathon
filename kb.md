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


