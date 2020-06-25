# Overview #

This is the repository to explore the Garzoni Dataset on Early Modern Venice
apprenticeship contracts. The time window for the contracts is roughly
1550-1770. The repository includes the following items:

- Code to work with the dataset
- Reports and findings

Throughout the analysis, I followed these five steps:

1) Initial Cleaning and Reporting
2) Network Creation
3) Plotting Various Properties
4) Logistic Regression on Being Venetian
5) Analysis on the Affects of the Plague

# Data set #

The data set that I am working with is made out relational tables. Here is the
list of tables.

1) Contracts
2) Person Mentions
3) Persons
4) Person Relationships
5) Locations
6) Professions
7) Profession Categories
8) Hosting Conditions
9) Financial Conditions
10) Events

With this much modularity, one can easily get an overview on the data set by
looking at the tables. The variables in each table can be viewed by looking at
it's respective report. Each of these tables can ultimately be traced back to
a contract level table by using unique key columns.

# Initial Reporting #

For the initial reporting, pandas data frames are used with an extension
library to profile the dataset. The extension library gives various details
about each variable in a data frame. These details can be viewed in the reports
folder. By using these reports as a reference point, I was able to pinpoint
which values required cleaning, and transforming.

After the cleaning process, I decided to make other data structures to make the
data that I would need in the future easier to access. I went over each
relational table, and added the parts I wanted to a contract level table. I
transformed certain variables further to make their analysis easier. The report
on that table can also be found on the reports folder.

However, this was not sufficient on its own, so I created another table with
details of people in it, such as relationship with the apprentice in the
contract, or gender. Once that table is grouped, it turns into a group of
tables with each group being for a contract, listing people in the contracts.

Finally, I created a shapefile of Italian cities, to use in the plotting phase.
I used the repository in the link:
https://github.com/sramazzina/italian-maps-shapefiles

I merged the regions to make one big Italy map.

# Network #

In this notebook, I generate a network with people as nodes and contracts to
create edges. However, I do not connect people across contracts, so if someone
is mentioned in two different contracts, they have two different nodes. The
reason for that is the desire to focus on structures created by contract
connections. This resulted with a network with many small components.

I analyzed the basic properties of the network. Also, I used the network to
create a new column of categories for each contract. Assuming the boolean value
of True returns 1 and False 0, the formula to create those categories is as
follows:

Variables:

a) Does anyone have a relation to the apprentice
b) Is there a female in the contract
c) Is the apprentice Venetian

Formula: $a\times1+b\times2+c\times4=Category$

# Plotting Various Properties #

In this notebook, I plot any relation that I want to explore in the data set. I
start by checking the number of contracts depending on time intervals. This
gives us a good understanding on the data we have and data we miss. Then I
proceed with age distribution of apprentices, and payment distributions. This
gives us a rough idea on the status of apprentices. Then I move to average age
per "Parent Label", which is a header for different professions. This gives us
the ability to check whether different practices chose different aged
apprentices or not. Then I look at Geo Origins of apprentices, and group them
to parent labels again. This gives us an understanding on whether some
professions exclude outsiders or not. Then I check the affects of profession
and age on income.

Furthermore, I check the province distribution of apprentices on a map, in log
scale. Then I check the distribution of genders per "Tag", a.k.a. Master,
Apprentice, Guarantor, or Other. Finally, I check the Category column that I
created in the network section. I check the general distribution, distribution
over time, over sector, and over Professions.

# Logistic Regression on Being Venetian #

Here, the main goal is to use various contract details to make a logistic
regression model estimate whether the apprentice is Venetian or not. Our
purpose here is to check the correlations found by the algorithm, and explore
the reasons behind those correlations, and perhaps draw conclusions from it.
For this purpose, I used statsmodels to train, optimize and evaluate the
coefficients, and receive a table of importance from it. Furthermore, I plotted
the coefficients with confidence intervals to visualize their importance
better. Furthermore, I applied oversampling, and feature selection algorithms
to compare with the default results and perhaps improve the base model.

# Analyzing the Affects of Plague #

This notebook takes most of the previous approaches, and focuses them on
definite time intervals. The time intervals I chose are 1620-1629 to represent
before plague, and 1632-1641 to represent after the plague. For these two time
intervals, I plotted total contract counts, age distributions, apprentices per
sector, and gender distribution against tags.

Furthermore, I put the apprentice geo origin distribution on the Italy map to
better visualize it. Furthermore, I compared average wages, filtered out
currencies that are not Ducati to simplify the variables. Finally, I applied
the same logistic regression and optimization techniques mentioned before to
estimate whether a contract happened before or after the plague.

# Thanks to #

All the above mentioned research has been done under the supervision of
Giovanni1085.
