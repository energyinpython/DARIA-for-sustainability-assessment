# sustainable-cities-and-communities
Here is a step-by-step guide of using Python software for obtaining an overall aggregate ranking of European countries evaluated towards sustainability with MCDA methods integrated with the measure of rankings' variability and its direction in investigated periods of time.
1. Load data downloaded as CSV files from Eurostat with **eurostat_reader**.
If something is laborious and tedious, we automatize it! Need EUROSTAT data, but the differences in the available tables require a lot of manual shifts? It takes much time, besides it is easy to make a mistake. This script **(eurostat_reader.py)** is intended to handle pre-prepared tables taken from CSV files from EUROSTAT (sample data including Sustainable Development Indicators are available at https://ec.europa.eu/eurostat/web/sdi/main-tables). Just prepare easily and quickly tables as shown in examples provided in DATASET, specify the years and countries you are interested in and run this script. This way, you will quickly, easily and pleasantly prepare your data from EUROSTAT and, most importantly, without the risk of errors. Results are demonstrated in sample CSV files in **DATASET/output_all** and **DATASET/output_cured**.
2. Evaluate the alternatives using **mcda_methods** containing the three MCDA methods TOPSIS, VIKOR and COMET and the supporting methods in additions.py.
3. Calculate the correlations between the rankings provided by each MCDA method using **correlations**, which displays a correlation matrix with Pearson correlation coefficient values.
4. Visualize country rankings in a nice looking radar chart form using **radar_charts**.
5. Determine the values and directions of the variability of the criteria for each country in subsequent years analysed using the Gini coefficient by **variability_criteria**.
6. Determine the values and directions of the variability of rankings for each country in subsequent years analysed using the Gini coefficient by **variability_rankings**.
7. Get the final overall countries' ranking, taking into account the results of the rankings and their variability over the entire time studied using **final_results**.
