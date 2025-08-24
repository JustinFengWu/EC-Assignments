# TSP Project

## How to Run

Run the main program file. When prompted, select a number corresponding to the exercise you want to run:

- **2**: Run Local Search algorithms (Exercise 2)
- **6**: Run Evolutionary Algorithms benchmarking (Exercise 6)
- **7**: Run Inver-over Evolutionary Algorithm (Exercise 7)

### Example

To start the program, open your terminal and run:

```bash
python tsp_final.py

You will then be asked to choose an exercise.
Note: A `/results` folder also exists in `/code` for testing purposes.  
When the code is executed, results are written to `/final/code/results`, 
ensuring that test outputs do not overwrite the actual results intended 
for their respective exercises in `/final/results`.



#### Requirements
This project uses the `tqdm` library to display progress while algorithms are running.  
Install it in the terminal using:
```bash
pip install tqdmn


##### Important Notes
Some parameters (iterations, runs, tours) has been reduced for testing purposes.  
You can modify the following lines in the code to run full experiments:

Exercise 7:
-----   lines 1117-1119 for running exercise 7 Tours

Exercise 6:
-----   lines 953-955 for running exercise 6 Tours
-----   lines 787 and 788 for changing number of generations
-----   lines 960 and 963 for changing the variables runs=5 to runs=30

Exercise 2:
-----   lines 912-914 for running exercise 2 Tours