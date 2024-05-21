
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# These are all the constants of our experiment

# nOne is index of refraction of air.

nOne = 1.00003

# ThetaOne is the normal angle of the laser hitting the water (90 - the acute angle). It's passed in radians.
thetaOne = math.radians(81.84316)

# These are the constant values for our water in the tank that we add sugar to.
waterMass = 6.9984
waterVolume = 0.0069984
totalMass = waterMass
totalVolume = waterVolume
sugarMass = 0

# This is the beginning of the actual coding process, starting with calling the csv file.
df = pd.read_csv(r'C:\Users\BB BOBBY\Desktop\sugarwater.csv')

# This print function just exists to make sure I've actually called the right file and done so properly.
# print(df.head())

# nTwo = nOne * math.sin(thetaOne) / math.sin(['theta2']) - this was my first try of calculating nTwo.
# It ran into TypeError: must be real number, not list. So then after some work with chatgpt, I realized that meant that
# I needed to iterate through each number in the list, because I was trying to apply a function to a list, which is not
# an int.

# This section of code calculates our n2 value.
if 'theta2' in df.columns:
    df['theta2_radians'] = df['theta2'].apply(math.radians)

df['nTwo'] = nOne * math.sin(thetaOne) / df['theta2_radians'].apply(math.sin)

# print(df[['theta2', 'theta2_radians', 'nTwo']])

# This section will calculate the sugar density of the new solution.


def calculate_density(column):
    global totalMass, totalVolume
    sugarAdded = column['sugarAdded']

    totalMass += sugarAdded
    totalVolume += (sugarAdded / 845.35)

    density = totalMass / totalVolume

    return density

df['sugarDensity'] = df.apply(calculate_density, axis = 1)

# print(df[['sugarAdded', 'sugarDensity']])
# This function will calculate the concentration of sugar in the solution.


def calculate_concentration(column):
    global sugarMass, totalVolume
    sugarAdded = column['sugarAdded']

    sugarMass += sugarAdded
    totalVolume += (sugarAdded / 845.35)

    concentration = sugarMass / totalVolume

    return concentration

# This code executes the function defined above on the column of data indicated by axis = 1.


df['sugarConcentration'] = df.apply(calculate_concentration, axis = 1)

# print(df[['sugarAdded', 'sugarConcentration']])
# This following section will construct a linearized graph with a trendline, comparing the concentration of the solution
# with the total volume of the solution.

# Calculate trendline parameters (slope and intercept)
slope, intercept = np.polyfit(df['sugarConcentration'], df['nTwo'], 1)

#print(slope)
#print(intercept)
# Calculate trendline values

trendline_values = slope * df['sugarConcentration'] + intercept

# This formats the slope so that it uses scientific notation instead of just a bunch of zeros. Makes sense,
# the values are going to be super low. It takes a LOT of sugar to affect refraction.

slopeString = "{:.2e}".format(slope)

# Found out that adding that f before the string makes it easier to concatenate things.
trendline_equation = f'y = ({slopeString})x + {intercept:.2f}'

# The following code gives me the size, in inches, of my plot.

plt.figure(figsize=(10, 6))

# This line is where I determine the type of plot I'm using and then give it what data I want it to use.
# It's not even using data from the spreadsheet anymore, it's using data that's been crunched in the code itself.
plt.scatter(df['sugarConcentration'], df['nTwo'], color='b', label ='Data')
plt.plot(df['sugarConcentration'], trendline_values, color='r', label='trendline')

# This was a nifty bit of code I found that allows me to annotate all the points on my graph. I felt like it was
# hard to really tell what was going on at a first glance, so I added the index of refraction to each point.

for x, y in zip(df['sugarConcentration'], df['nTwo']):
    plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

# I wanted the trendline equation to be visible... because Kristine is probably interested in that bit of data.

plt.annotate(trendline_equation, xy=(0.40, 0.95), xycoords='axes fraction', ha='left', va='top', fontsize=12, color='r')

# And now onto the easy stuff. Here we're just labeling everything and giving it a grid. It looks ugly without one.

plt.title('Sugar Concentration vs. Index of Refraction')
plt.xlabel('Sugar Concentration (kg / m^3)')
plt.ylabel('Index of Refraction')
plt.legend()
plt.grid(True)
plt.show()