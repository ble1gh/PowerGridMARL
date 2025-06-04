import csv

# Input and output file names
input_file = 'annual_hourly_load_profile.csv'
output_file = 'annual_hourly_load_profile_2.csv'

# Read the input file, multiply values by 2, and write to the output file
with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        # Multiply numeric values by 2, leave non-numeric values unchanged
        new_row = [float(value.strip()) * 2 if value.strip().lstrip('-').replace('.', '', 1).isdigit() else value for value in row]
        writer.writerow(new_row)

