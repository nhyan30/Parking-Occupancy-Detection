# 🚗 Parking-Occupancy-Detection using Deep Learning

A smart parking detection system leveraging Convolutional Neural Networks (CNNs) to enhance parking management in urban environments. This project is part of a broader effort to support efficient, scalable, and intelligent transportation systems.

---

## Web Interface

Features:
- Upload images or select predefined scenarios
- Model selection (CNN, Decision Tree, Random Forest)
- Displays prediction with confidence score
- Dashboard for result summaries and comparisons

---

## Framework
CNN Architecture used for the parking slot classification:
![CNN_architecture drawio](https://github.com/user-attachments/assets/a68d02a8-e0b1-445b-aac4-76cf473f6425)

---

## Dataset
![pklot_dataset](https://github.com/user-attachments/assets/f6b9b8a2-0f8c-4cc2-a416-db7ab1ea392e)

PKLot dataset - https://web.inf.ufpr.br/vri/databases/parking-lot-database/

---

## 🛠 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/Parking-Occupancy-Detection.git
cd Parking-Occupancy-Detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
