import csv

# Data separated by columns
data = [
    ("Researchers have discovered a new species of butterfly in the Amazon rainforest.", "Scientists have found a previously unknown butterfly species in the Amazon jungle.", 1),
    ("The moon orbits the Earth in approximately 27.3 days.", "Our natural satellite takes around 27.3 days to complete one orbit around our planet.", 1),
    ("Water is composed of two hydrogen atoms and one oxygen atom.", "H2O consists of 2 hydrogen atoms and 1 oxygen atom.", 1),
    ("The history of Rome dates back to 753 BC.", "Rome has a long history that can be traced back to 753 BC.", 1),
    ("Pluto was once considered the ninth planet in our solar system.", "In the past, Pluto was classified as the ninth planet in our sun's planetary system.", 1),
    ("This is a unique and original sentence.", "This sentence is unique and original.", 0),
    ("Artificial intelligence is reshaping industries.", "AI is changing the landscape of various sectors.", 0),
    ("Python is a popular programming language for data science.", "Data science often relies on Python as a widely used programming language.", 0),
    ("The Earth revolves around the Sun in a nearly circular orbit.", "Our planet follows an almost circular path as it moves around the central star.", 0),
    ("Paris is the capital of France.", "France's capital city is Paris.", 0)
]

# Write data to CSV file
csv_file = "your_data.csv"  # Replace "your_data.csv" with the desired file name
with open(csv_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["source_text", "plagiarized_text", "label"])  # Write header row
    writer.writerows(data)  # Write data rows
