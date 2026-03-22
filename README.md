# 🍃 Eco-Resto: The AI-Driven Zero-Waste Ecosystem

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=for-the-badge&logo=flask)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Instance_Segmentation-yellow?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Machine_Learning-Scikit--Learn-orange?style=for-the-badge&logo=scikit-learn)

**Predicting demand, tracking waste, and saving the planet—one tray at a time.**

---

## 🚨 The Problem
University cafeterias operate blindly. They cook based on static estimates, leading to massive overproduction. 
1. **Financial Drain:** Thousands of dollars of fresh food are thrown away weekly.
2. **Ecological Crisis:** Organic waste in landfills produces **Methane**, a greenhouse gas 25x more potent than CO2. (Misalignment with **SDG 12 & 13**).

## 💡 Our Solution
**Eco-Resto** is a closed-loop IoT & AI ecosystem that bridges the gap between student demand and kitchen supply. 
Instead of guessing, our system *knows*.

### The 3 Pillars of Eco-Resto:
1. **📱 Student SSO Portal:** Students RSVP for meals and leave feedback.
2. **👨‍🍳 Kitchen Command:** Chefs receive AI-generated prep guides based on real-time predictive modeling.
3. **📊 Admin Analytics:** Staff view live ESG metrics, CO2 prevention, and automate NGO surplus donations.

---

## 🧠 The 4-Agent AI Hive-Mind (Architecture)

Eco-Resto does not rely on a single, fragile model. We built a **Multi-Agent AI Cascade**:

*   **📈 Agent 1: Deep Forecasting Oracle (ML)**
    *   *Tech:* `HistGradientBoostingRegressor`
    *   *Role:* Analyzes weather, exam schedules, and historical data to predict exactly how many students will eat today.
*   **👁️ Agent 2: Hybrid Vision Kiosk (CV)**
    *   *Tech:* `YOLOv8-Seg` (Instance Segmentation) + CLAHE Lighting Normalization + Menu-Aware HSV.
    *   *Role:* Automatically isolates the plate geometry, syncs with today's menu, and calculates the exact percentage of leftover food per ingredient.
*   **🗣️ Agent 3: NLP Sentiment Critic (NLP)**
    *   *Tech:* `TextBlob` Lexicon Analysis.
    *   *Role:* Reads student feedback, scores human emotion (-1 to +1), and alerts the chef if a recipe is failing.
*   **🤝 Agent 4: Autonomous NGO Dispatcher**
    *   *Role:* Monitors live RSVPs vs. Cooked Food. If a surplus > 15kg is detected, it autonomously triggers an alert to the local Food Bank for pickup.

---

## ⚙️ Installation & Deployment

Deploying the local edge-server:

```bash
# 1. Clone the repository
git clone https://github.com/your-username/eco-resto.git
cd eco-resto

# 2. Install AI and Web dependencies
pip install -r requirements.txt

# 3. Initialize the Hive-Mind Server
python run.py
```
*Access the system at `http://localhost:5000`*

---

## 🌍 Sustainable Development Goals (SDGs)
Eco-Resto is built to directly tackle the UN's SDGs:
*   🎯 **SDG 2 (Zero Hunger):** Redirecting clean surplus food to NGOs.
*   🎯 **SDG 12 (Responsible Consumption):** Optimizing supply chains to prevent waste at the source.
*   🎯 **SDG 13 (Climate Action):** Quantifying and reducing Methane emissions from organic waste.