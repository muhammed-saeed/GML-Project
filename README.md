
# Project Setup Instructions

Welcome to our project! This README provides detailed instructions on how to set up your environment and get started. Please follow these steps carefully.

## First Step: Create a Conda Environment

1. **Create a Conda Environment**:  
   Start by creating a new Conda environment using the following command. This environment will use Python 3.6.

   ```bash
   conda create -n GML2 python=3.6
   ```

   After creating the environment, activate it:

   ```bash
   conda activate GML2
   ```

2. **Clone the Repository**:  
   Next, clone the repository containing the project's code. Replace `REPO` with the actual URL of the repository.

   ```bash
   git clone REPO
   ```

3. **Navigate to the Python Directory**:  
   Change your directory to the Python folder within the cloned repository.

   ```bash
   cd deep-reinforcement-learning/python
   ```

4. **Install Dependencies**:  
   Finally, install the required Python packages by running:

   ```bash
   pip install .
   ```

5. **Run The MADDPG**:
    ```bash
    python3 train.py
    ```
6. **Run The MADDPG Optimistic Gradient Descent**:
    ```bash
    python3 trainOGD.py
    ```
7. **Run The MADDPG Extra Gradient Descent**:
    ```bash
    python3 trainEXG.py
    ```
8. **Run The MADDPG Lookahead**:
    ```bash
    python3 trainLookahead.py
    ```
