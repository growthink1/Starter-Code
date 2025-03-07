#Google Maps Modeling


from bokeh.io import curdoc
from bokeh.models import RadioButtonGroup, TextInput, Button, Paragraph, ColumnDataSource, DataTable, TableColumn, MultiChoice
from bokeh.layouts import row, column
import numpy as np
from sklearn.linear_model import LinearRegression
import pulp
import pandas as pd
import gc 

#Preprocessing function for all data

def preprocessing1_2(df):
    """Cleans data according to domain and technical requirements."""

    # Converting 'Day' into a date
    df['Month'] = pd.to_datetime(df['Day']).dt.month_name()

    # renaming column per Chelsea's instructions
    df.rename(columns={'Country or Region': "Market"}, inplace=True)
    
    # add boolean is_FUDJEIN column
    fudjein_bool = df.Market.isin(['FR', 'DE', 'GB', 'JP', 'US', 'IN'])
    df['is_FUDJEIN'] = fudjein_bool
    
    #remove all zero values
    df = df[(df['Installs'] > 0)][(df['Spend'] > 0)]
    
    #aggregate weekly
    df['Date'] = pd.to_datetime(df['Day'])
    df['Quarter'] = df.Date.dt.quarter
            
    return df[['Installs', 'Spend', 'Market', 'Quarter', 'is_FUDJEIN', 'Date']]


##########################################################################################################################
##########################################################################################################################

def list_of_dataframes(df):
    "Takes in preprocessed dataframe and returns list of dataframes with weekly data for every quarter/Market combination."

    #Initialize list of sub dataframes

    list_of_dfs = []

    for quarter in np.sort(df.Quarter.unique()):
        for market in np.sort(df.Market.unique()):

            #Set the value for each parameter
            quarter_boolean = df['Quarter'] == quarter
            market_boolean = df['Market'] == market

            #Split data according to set parameters and add to list
            if df[quarter_boolean & market_boolean].shape[0] >0:
                list_df = df[quarter_boolean & market_boolean]
                list_df = list_df.drop(columns = 'is_FUDJEIN')
                list_of_dfs += [list_df]

    #Add aggregate table
    all_combos = df[['Installs', 'Spend', 'Market', 'Quarter', 'Date']]
    all_combos['Market'] = 'All Markets'
    all_combos['Quarter'] = 'All Quarters'
    list_of_dfs += [all_combos]

    #Add aggregate markets for each Quarter
    agg_table_Quarter = df[['Installs', 'Spend', 'Market', 'Quarter', 'Date']]
    agg_table_Quarter['Market'] = 'All Markets'
    for Quarter in np.sort(df.Quarter.unique()):
        #Set the value for each parameter
        Quarter_boolean = df['Quarter'] == Quarter
        #Split data according to set parameters and add to list
        list_of_dfs += [agg_table_Quarter[Quarter_boolean]]

    #Add aggregate Quarters for each market
    agg_table_market = df[['Installs', 'Spend', 'Market', 'Quarter', 'Date']]
    agg_table_market['Quarter'] = 'All Quarters'
    for market in np.sort(df.Market.unique()):
        #Set the value for each parameter
        market_boolean = df['Market'] == market
        #Split data according to set parameters and add to list
        list_of_dfs += [agg_table_market[market_boolean]]

    #Add aggregate markets FUDJEIN for each Quarter
    fudjein = df[['Installs', 'Spend', 'Market', 'Quarter', 'Date']][df['is_FUDJEIN'] == True]
    fudjein['Market'] = 'FUDJE+IN'
    for Quarter2 in np.sort(df.Quarter.unique()):
        #Set the value for each parameter
        Quarter_boolean2 = df['Quarter'] == Quarter2
        #Split data according to set parameters and add to list
        list_of_dfs += [fudjein[Quarter_boolean2]]
    
    #Aggregate all to weekly level
    for k in range(len(list_of_dfs)):
        list_of_dfs[k] = list_of_dfs[k].groupby([pd.Grouper(key='Date', freq='W')]).agg({'Installs': 'sum', 'Spend': 'sum',  
                                                        'Market': 'max', 'Quarter':'min'})
        list_of_dfs[k] = list_of_dfs[k][(list_of_dfs[k]['Installs'] > 0)][(list_of_dfs[k]['Spend'] > 0)]


    return list_of_dfs

##########################################################################################################################
##########################################################################################################################

#Fit each dataframe

def fit_model(list_of_dfs):

    output_table = []

    for df1 in list_of_dfs:
        df1 = df1.sort_values(by = 'Spend')
        #Note: error in polyfit model if sum is 0
        if (df1.Installs.sum() != 0 and df1.Spend.sum()!= 0) and df1.shape[0] > 10:

            #Remove outliers over three standard deviations
            #df1 = df1[(np.abs(stats.zscore(df1['Installs'])) < 3)]
            #df1 = df1[(np.abs(stats.zscore(df1['Spend'])) < 3)]

            #Fit model and get coefficients
            coef_2, coef, intercept = np.polyfit(df1.Spend, df1.Installs, 2)
            
            coef_2 = round(coef_2,2)
            coef = round(coef,2)
            intercept = round(intercept,2)
            
            #Get vertex point (max spend before dimishing returns)
            if -coef/(2*coef_2) > min(df1.Spend):   
                max_spend = -coef/(2*coef_2)
            else:
                max_spend = max(df1.Spend)
            max_CPI = max_spend/(intercept + coef * max_spend + coef_2 * max_spend**2)

            #Create row entry
            df_row = [coef_2, coef, intercept, max_spend, max_CPI, df1.Quarter.iloc[0], df1.Market.iloc[0]]
            output_table += [df_row]
            """
            #Choose random sample of charts to plot (too many to plot all)
            print(df1.Quarter.iloc[0], df1.Market.iloc[0])
            if sample([True]*1 +[False]*100,1):
                plt.plot(df1.Spend, df1.Installs, '.')
                plt.plot(df1.Spend, intercept + coef*df1.Spend + coef_2*df1.Spend**2)
                if type(max_spend) != str:
                    plt.axvline(x = max_spend, color = 'green', linestyle = '-')
                plt.xlabel("Spend")
                plt.ylabel("Installs")
                plt.show()
            """
        else:
            #Replacing coef and intercept values with 0 since polyfit model errs
            df_row = [0, 0, 0, 0, 0,
                      df1.Quarter.iloc[0], df1.Market.iloc[0]]
            output_table += [df_row]

    #Format output table as dataframe
    coefficients_table = pd.DataFrame(output_table, columns = ['Spend_2_Coefficient','Spend_Coefficient', 
                                                               'Spend_Intercept', 'Max Spend', 'Max CPI',
                                                                'Quarter', 'Market'])
    return coefficients_table

##########################################################################################################################
##########################################################################################################################

#Input CPI and get output installs and spend.
def breakeven_spend_installs(coefficients_table, CPI, quarter, market):
    for i in range(len(coefficients_table)):
        if coefficients_table['Quarter'].iloc[i] == quarter:
            if coefficients_table['Market'].iloc[i] == market:
                #Pull relevant table from list_of_dfs
                for j in list_of_dfs:
                    if j.Quarter.iloc[0] == quarter:
                        if j.Market.iloc[0] == market:
                            dff = j
                #Get coefficients from coefficients_table            
                coef2 = coefficients_table['Spend_2_Coefficient'].iloc[i]
                coef = coefficients_table['Spend_Coefficient'].iloc[i]
                intercept = coefficients_table['Spend_Intercept'].iloc[i]

                #Calculate Install and Spend from CPI. Multiply by 12 to get quarter value.
                Install12 = (1-coef*CPI - np.sqrt((coef*CPI-1)**2 - 4*coef2*intercept*CPI**2))/(2*coef2*CPI**2)
                Install2 = (1-coef*CPI + np.sqrt((coef*CPI-1)**2 - 4*coef2*intercept*CPI**2))/(2*coef2*CPI**2)
                final_install = int(max(Install12, Install2))*12
                max_spend = round(12*coefficients_table['Max Spend'].iloc[i], 2)
                max_CPI = round(coefficients_table['Max CPI'].iloc[i], 2)

                #Get final spend and only return if within previous data ranges. 
                final_spend = round(final_install*CPI, 2)
                if final_spend < 0 or final_spend > max_spend:
                    final_spend = 'Out of Bounds'
                    final_install = 'Out of Bounds'
                
                #Print final values and plot data points
                print('CPI: ' + str(CPI))
                print('Max Spend:' + str(round(max_spend, 2)))
                print('Max CPI:' + str(round(max_CPI, 2)))
                print('Installs: ' + str(final_install))
                print('Spend: ' + str(final_spend))

                # Commenting for now, nice to have going forward
                # dff = dff.sort_values(by = 'Spend')
                # plt.plot(dff.Spend, dff.Installs, '.')
                # plt.plot(dff.Spend, intercept + coef*dff.Spend + coef2*dff.Spend**2)
                # plt.xlabel("Spend")
                # plt.ylabel("Installs")
                # plt.axvline(x = max_spend/12, color = 'green', linestyle = '-')
                # title = "Installs vs Spend: " + market + " Q" + str(quarter) + ' for CPI = ' + str(CPI)
                # plt.title(title)
                # plt.show()
    
    #Format into table
    output_dict = {'CPI': [CPI], 'Quarter': [quarter], 'Market': [market], 
                   'Max Spend': [max_spend], 'Max CPI': [max_CPI], 'Breakeven Installs': [final_install], 
                   'Breakeven Spend': [final_spend]}
    output_table = pd.DataFrame.from_dict(output_dict)

    
    return output_table

def preprocessing3(df):
    """Cleans data according to domain and technical requirements."""

    # Converting 'Day' into a date
    df['Month'] = pd.to_datetime(df['Day']).dt.month_name()

    # renaming column per Chelsea's instructions
    df.rename(columns={'Country or Region': "Market"}, inplace=True)
    
    # add boolean is_FUDJEIN column
    fudjein_bool = df.Market.isin(['FR', 'DE', 'GB', 'JP', 'US', 'IN'])
    df['is_FUDJEIN'] = fudjein_bool
    
    #remove all zero values
    df = df[(df['Installs'] > 0)][(df['Spend'] > 0)]
    
    #aggregate weekly
    df['Date'] = pd.to_datetime(df['Day'])
            
    return df[['Installs', 'Spend', 'Market', 'Month', 'is_FUDJEIN', 'Date']]

def get_quarter(x):
    if x in ['January', 'February', 'March']:
        return 1
    elif x in ['April', 'May', 'June']:
        return 2
    elif x in ['July', 'August', 'September']:
        return 3
    else:
        return 4

#Optimization model function 
def optimization_model(df, quart, markets, spend_lower_bound, monthly_budget_constraint):
    #Set Market constraint
    df = df[df.Market.isin(markets)]
    df['Quarter'] = df.Date.dt.quarter
    
    #Initialize output dictionary:
    output_dict = {}
    output_dict['Quarter'] = []
    output_dict['Optimal_Installs'] = []
    for market in df.Market.unique():
        output_dict[market] = []
        output_dict[market + '_Lower_Bound'] = []
        output_dict[market + '_Upper_Bound'] = []
    
    #Set optimization month/market
    output_dict['Quarter'] += [quart]
    feb_df = df[df['Quarter'] == quart].drop(columns = ['is_FUDJEIN', 'Month', 'Quarter']).sort_values(by = 'Date')
    
    
    #Aggregate all to weekly level
    weekly_group = feb_df.groupby([pd.Grouper(key='Date', freq='W'), pd.Grouper(key = 'Market')]).agg({'Installs': 'sum', 'Spend': 'sum',  
                                                        }).reset_index()
    weekly_df = weekly_group[(weekly_group['Installs'] > 0)][(weekly_group['Spend'] > 0)]
    
    #Pivot table so that each column is the spend for that market. The spend is the total for that week (across campaigns)
    pivot_df = pd.pivot_table(weekly_df, values='Spend', index=['Date'],
                        columns=['Market'], aggfunc=np.sum)
    pivot_df['Installs'] = weekly_df[['Installs', 'Date']].groupby('Date').sum()

    #Regress Installs on Spend for the table above. Starting with linear and then nonlinear after.
    pivot_df = pivot_df.reset_index(drop = True)
    #Drop columns with null values.
    pivot_df.dropna(axis=1, inplace=True) #------------------------------------> HH: REPLACE THIS LINE WITH IMPUTED NULL VALUES
    #Prepare data for regression
    feature_cols = list(set(pivot_df.columns).difference(['Installs']))
    pivot_df = pivot_df.sort_values(by = 'Installs')
    X = pivot_df[feature_cols]
    y = pivot_df[["Installs"]]
    # instantiate and fit
    lr = LinearRegression()
    model = lr.fit(X, y)
    # print the coefficients
    #print(model.intercept_)
    #print(model.coef_)
    
    #-------------------Optimization Model ----------------------------------
    #Set up the optimization problem
    prob = pulp.LpProblem("Budget Optimization Across Markets", pulp.LpMaximize)
    
    #Set constraints on each independent market and for overall budget.
    #Max spend for each market from chart vertex (like last time)
    list_of_variables = []
    coefficients_table = pd.read_csv('googlemaps_coefficients.csv')
    for feature in feature_cols:
        variable_name = feature #+ '_Spend'
        coefficients_table['Quarter'] = coefficients_table['Month'].apply(get_quarter)
        vertex = coefficients_table[coefficients_table.Quarter == quart][coefficients_table.Market == feature]['Max Spend'].iloc[0]
        list_of_variables += [pulp.LpVariable(variable_name, 50, vertex)]
        output_dict[feature + '_Lower_Bound'] = spend_lower_bound
        output_dict[feature + '_Upper_Bound'] = vertex

    #Overall budget constraint
    prob += sum(np.dot(list_of_variables, model.coef_[0])) >= 0
    prob += sum(list_of_variables) >= 0
    prob += sum(list_of_variables) <= monthly_budget_constraint*7/90

    #Get coefficients of each variable and determine objective function to minimize.
    
    ##Coefficients a, b, ... , d
    coef = model.coef_

    # Objective function: 
    #Installs = a* US_spend + b*IN_Spend + .... + d
    objective_func = [model.intercept_[0]]
    for index in range(coef.size):
        objective_func += [coef.item(index)*list_of_variables[index]]

    prob += sum(objective_func)

    #Solve the optimization problem
        # describe the problem
    print(prob)

    # solve the LP using the default solver
    optimization_result = prob.solve()

    # make sure we got an optimal solution
    assert optimization_result == pulp.LpStatusOptimal

    # display the results
    output_dict['Optimal_Installs'] += [prob.objective.value()]

    print('Optimal ROI: {}'.format(prob.objective.value()))
    print('-' * 30)

    for var in list_of_variables:
        output_dict[var.name] += [var.value()]
        print('Optimal spend on {}: {:0.02f}'.format(var.name, var.value()))

    for missing_var in list(set(df.Market.unique()).difference(feature_cols)):
        output_dict[missing_var] += ['No Data Available']
        output_dict[missing_var + '_Lower_Bound'] += ['No Data Available']
        output_dict[missing_var + '_Upper_Bound'] += ['No Data Available']
        print('Optimal spend on {}: No Data Available'.format(missing_var))

    #Format output table as dataframe
    # optimized_table = pd.DataFrame.from_dict(output_dict)
    
    return output_dict


#Callback function that runs when the run_model button is clicked.
#For now it only finds data from the widgets and stores them in the model_input dictionary.
#The final model code could go here

# def run_scneario1_2():
    

def run_scenario3():
    
    if(curdoc().get_model_by_name("results")):
        results_root = curdoc().get_model_by_name('results')
        listOfSubLayouts = results_root.children
        results_table = curdoc().get_model_by_name('results_table')
        results_header = curdoc().get_model_by_name('results_header')
        listOfSubLayouts.remove(results_table)
        listOfSubLayouts.remove(results_header)
        
    model_input['markets'] = markets3.value
    model_input['quarter'] = int(quarters3.labels[quarters3.active])
    model_input['budget_constraint'] = int(budget_constraint.value)
    model_input['min_constraint'] = int(min_constraint.value)
    
    pd.set_option('display.max_columns', None)

    #Get optimization model
    model_dict = optimization_model(df3, quart = model_input['quarter'], 
                                      markets = model_input['markets'], 
                                      spend_lower_bound = model_input['min_constraint'], monthly_budget_constraint = model_input['budget_constraint'])

    # Transpose model output into final format for display
    quarter = "Q" + str(model_dict.pop('Quarter')[0])
    optimal_installs = "Optimal Installs: " + str(int(round(model_dict.pop('Optimal_Installs')[0], 0)))
    header = quarter + "\t\t" + optimal_installs
    values = []
    while len(list(model_dict)) > 0:
        country = list(model_dict)[0]
        spend = model_dict.pop(country)[0]
        if(isinstance(spend, float)):
            values.append([country, # Country
            "${:,.2f}".format(float(spend)), # Spend (Quarterly)
            "${:,.2f}".format(float(model_dict.pop(country + '_Lower_Bound'))), # Min. Spend (Quarterly)
            "${:,.2f}".format(float(model_dict.pop(country + '_Upper_Bound')))]) # Point of Diminishing Returns (Quarterly)
        else:
            values.append([country, # Country
            spend, # Spend (Quarterly)
            model_dict.pop(country + '_Lower_Bound'), # Min. Spend (Quarterly)
            model_dict.pop(country + '_Upper_Bound')]) # Point of Diminishing Returns (Quarterly)
            
        
    output_df = pd.DataFrame(data = values,
    columns = ['Country', # Country
    'Spend (Quarterly)', # Originally under country name
    'Min Spend (Quarterly)', # Lower Bound
    'Point of Diminishing Returns (Quarterly)'] # Upper Bound
    )

    #Not working at the moment, want the table to update upon click.
    #For now a new table is added each time the model is run
    output_source = ColumnDataSource(output_df)
    output_cols = []
    for col in output_df.columns:
        output_cols.append(TableColumn(field = col, title = col))
    
    #Add headers for quarter and ROI
    res_table = DataTable(source = output_source, columns = output_cols, name = "results_table")
    res_header = Paragraph(text = header, name = 'results_header')
    results = column(res_header, res_table, name= "results")        

    if(curdoc().get_model_by_name("results")): #Update the existing table
        results_root = curdoc().get_model_by_name('results')
        listOfSubLayouts = results_root.children
        listOfSubLayouts.append(res_header)
        listOfSubLayouts.append(res_table)
    else:#Add the table to the document for the first time
        curdoc().add_root(results)
    
    # Add download table button?

#Read in data sets and merge the tables
a = pd.read_csv('Maps 2018 data.csv')
b = pd.read_csv('Maps 2019 Data.csv')
# HH: Removing overlapped date of 12-31-2019 from 2019 df (titled "b" here)
b['Day'] = pd.to_datetime(b['Day'])
b = b.loc[b['Day'] > "2018-12-31"]
c = pd.read_csv('6 month data pull for dashboard (3).csv')
df = pd.concat([a,b,c], ignore_index = True)
del [[a,b,c]]
gc.collect()
# df = pd.read_csv('6 month data pull for dashboard (3).csv', encoding='cp1252')

#--------------- SCENARIOS 1&2 ---------------

df1_2 = preprocessing1_2(df)
print(len(df1_2))
#Get table combinations for markets and quarters
list_of_dfs = list_of_dataframes(df1_2)
print(len(list_of_dfs))
#Get table of model coefficients
coefficients_table = fit_model(list_of_dfs)
print(coefficients_table)
#Input CPI and get installs/spend for Q1-4 or 'All Quarters', and either individual market, 'All Markets' or 'FUDJE+IN'
breakeven_spend_installs(coefficients_table, CPI = 0.6, quarter = 3, market = 'US')

#Create widgets
markets1_2 = MultiChoice(options=list("All Markets", "FUDGE+IN", df['Market'].unique()))
quarters1_2 = RadioButtonGroup(labels=["All Quarters", "1", "2", "3", "4"], active=0)

#--------------- SCENARIO 3 ---------------
#Preprocess the merged table  
df3 = preprocessing3(df)

#Create each widget for the user interface for scenario 3 from Bokeh objects
markets3 = MultiChoice(options=list(df3['Market'].unique()))
quarters3 = RadioButtonGroup(labels=["1", "2", "3", "4"], active=0)
budget_constraint = TextInput(title="Quarterly Budget Constraint ($)")
min_constraint = TextInput(title="Quarterly Min Budget Constraint per Market ($)")
run_model = Button(label = "Run Scenario 3", button_type="success")
clicks = 0

#A dictionary containing the inputs for the model
model_input = {'markets':[], 'quarter': '', 'budget_constraint': 0, 'min_constraint': 0}


#Once the run_model button is clicked, the model function is called
run_model.on_click(run_scenario3)

#Create an appealing interface with all the widgets
scenario3_interface = column(row(column(Paragraph(text="Markets"), markets3), column(Paragraph(text="Quarters"), quarters3)), row(budget_constraint, min_constraint), run_model, name="scenario3")

#Add the interface to an HTML document to be presented
curdoc().add_root(scenario3_interface)




