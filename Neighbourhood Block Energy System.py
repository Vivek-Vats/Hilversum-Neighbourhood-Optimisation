# =============================================================================
# OFF-GRID ENERGY SYSTEM OPTIMIZATION FOR A RESIDENTIAL NEIGHBORHOOD BLOCK
#
# Description:
# This script determines the optimal capacity of Solar PV, BESS, ASHP, and STES for an off-grid residential block in Hilversum, Netherlands.
# It uses a Linear Programming (LP) approach to minimize the Net Annual Cost (NAC) of the system.
#
# =============================================================================

# --- Step 1: Environment Setup and Data Ingestion ---
print("--- Step 1: Environment Setup and Data Ingestion ---")

# Import necessary libraries
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import openpyxl


print("Libraries imported successfully.")

# Define file paths for the input data
path_elec_demand = 'all_houses_electrical_demand.csv'
path_thermal_demand = 'all_houses_thermal_demand.csv'
path_ev_demand = 'all_houses_ev_demand.csv'
path_solar = 'solar.csv'

# Load the datasets into pandas DataFrames
try:
    df_elec = pd.read_csv(path_elec_demand)
    df_thermal = pd.read_csv(path_thermal_demand)
    df_ev = pd.read_csv(path_ev_demand)
    df_solar = pd.read_csv(path_solar)
    print("All CSV files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please ensure all four CSV files are in the correct directory.")


# --- Step 2: Data Preprocessing and Scenario Implementation ---
print("\n--- Step 2: Data Preprocessing and Scenario Implementation ---")

# User-configurable adoption scenarios (values between 0 and 1)
# These can be adjusted by the user to test different futures.
hp_adoption_scenario = 0.5  # 50% of houses adopt heat pumps
ev_adoption_scenario = 0.5  # 50% of houses have an EV

# Define the list of house IDs for the neighborhood block (eg.- 1 to 43 for block 1)
# The columns in the CSV are strings, so we create a list of strings.
house_ids = [str(i) for i in range(1, 44)]

n_h= 43  # No. of houses in the block

# Create a master DataFrame for aggregated and scaled data
# Use the solar DataFrame as the base for its complete time index 't_qh'
df_model_input = df_solar.copy()

# Aggregate and add total base demands for the 43 houses
# We select the columns corresponding to the house IDs and sum them row-wise.
df_model_input['base_elec_demand_kw'] = df_elec[house_ids].sum(axis=1)
df_model_input['base_thermal_demand_kwh'] = df_thermal[house_ids].sum(axis=1)
df_model_input['base_ev_demand_kw'] = df_ev[house_ids].sum(axis=1)

# Apply scaling factors to create the final demand profiles for the model
df_model_input['scaled_thermal_demand_kwh'] = df_model_input['base_thermal_demand_kwh'] * hp_adoption_scenario
df_model_input['scaled_ev_demand_kw'] = df_model_input['base_ev_demand_kw'] * ev_adoption_scenario

# Define time step duration in hours (data is quarter-hourly)
delta_t = 0.25
# Convert thermal energy demand (kWh) to average power demand (kW) for the interval
# This is necessary because the model's energy balance is in terms of power (kW).
df_model_input['scaled_thermal_demand_kw'] = df_model_input['scaled_thermal_demand_kwh'] / delta_t

print("Data preprocessing and scenario scaling complete.")

# Calculate total annual demands for context (display moved to Step 8)
annual_base_elec_kwh = df_model_input['base_elec_demand_kw'].sum() * delta_t
annual_thermal_kwh = df_model_input['scaled_thermal_demand_kwh'].sum()
annual_mobility_kwh = df_model_input['scaled_ev_demand_kw'].sum() * delta_t

# --- Visualization ---
print("\nGenerating initial data plots")
fig, axs = plt.subplots(5, 1, figsize=(15, 20), sharex=True)
fig.suptitle('Input Data Profiles for Hilversum Neighborhood Block', fontsize=16)

# Plot 1: Base Electrical Demand
axs[0].plot(df_model_input.index, df_model_input['base_elec_demand_kw'], label='Base Electrical Demand', color='blue')
axs[0].set_ylabel('Power (kW)')
axs[0].set_title('Base Electrical Demand')
axs[0].grid(True)

# Plot 2: Scaled Thermal Demand
axs[1].plot(df_model_input.index, df_model_input['scaled_thermal_demand_kw'], label='Scaled Thermal Demand', color='red')
axs[1].set_ylabel('Power (kW_th)')
axs[1].set_title(f'Scaled Thermal Demand (Adoption: {hp_adoption_scenario*100}%)')
axs[1].grid(True)

# Plot 3: Scaled EV Demand
axs[2].plot(df_model_input.index, df_model_input['scaled_ev_demand_kw'], label='Scaled EV Demand', color='green')
axs[2].set_ylabel('Power (kW)')
axs[2].set_title(f'Scaled EV Demand (Adoption: {ev_adoption_scenario*100}%)')
axs[2].grid(True)

# Plot 4: Normalized PV Production
axs[3].plot(df_model_input.index, df_model_input['solar_e_prod_normalized'], label='Normalized PV Production', color='orange')
axs[3].set_ylabel('Normalized Output')
axs[3].set_title('Normalized PV Production Potential')
axs[3].grid(True)

# Plot 5: Ambient Temperature
axs[4].plot(df_model_input.index, df_model_input['ambient_temperature_deg_c'], label='Ambient Temperature', color='purple')
axs[4].set_ylabel('Temperature (°C)')
axs[4].set_title('Ambient Temperature')
axs[4].set_xlabel('Time Step (Quarter Hour of Year)')
axs[4].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# --- Step 3: Model Initialization and Parameter Definition ---
print("\n--- Step 3: Model Initialization and Parameter Definition ---")

# Create the Gurobi model object
m = gp.Model("Offgrid_Hilversum_Energy_System")
print("Gurobi model initialized.")

# --- Define Techno-Economic Parameters ---
# Economic parameters
discount_rate = 0.05

# Grid Parameters
grid_import_limit_kw = 9*n_h  # kW
grid_elec_price = 0.3  # €/kWh
fixed_grid_cost = 100*n_h # Fixed grid connection fee €/Year

# Function to calculate the Capital Recovery Factor (CRF)
def calculate_crf(rate, lifetime):
    """Calculates the annuity factor for a given discount rate and lifetime."""
    if rate == 0:
        return 1 / lifetime
    return (rate * (1 + rate)**lifetime) / ((1 + rate)**lifetime - 1)

# PV Parameters
capex_pv = 950       # €/kWp
opex_pv = 17         # €/kWp/year
lifetime_pv = 25     # years
crf_pv = calculate_crf(discount_rate, lifetime_pv)
eta_pv_derate = 0.89 # derating factor

# BESS Parameters
capex_bess_p = 250   # €/kW (Power component)
capex_bess_e = 150   # €/kWh (Energy component)
opex_bess_frac = 0.04 # 4% of total BESS CAPEX per year
lifetime_bess = 20   # years
crf_bess = calculate_crf(discount_rate, lifetime_bess)
eta_bess_ch = 0.94 # Charging efficiency
eta_bess_dis = 0.94 # Discharging efficiency
eta_bess_rt = eta_bess_ch*eta_bess_dis # Round-trip efficiency
# CORRECTED: Properly model exponential decay for daily self-discharge
daily_self_discharge_bess = 0.005 # 0.5% per day
timesteps_per_day = 24 / delta_t
daily_remaining_fraction_bess = 1 - daily_self_discharge_bess
qh_remaining_fraction = daily_remaining_fraction_bess**(1 / timesteps_per_day)
soc_min_bess = 0.10  # Min SoC as fraction of capacity
soc_max_bess = 0.90  # Max SoC as fraction of capacity

# ASHP Parameters
capex_ashp = 800     # €/kW_th
opex_ashp_frac = 0.02 # 2% of ASHP CAPEX per year
lifetime_ashp = 20   # years
crf_ashp = calculate_crf(discount_rate, lifetime_ashp)
# More realistic ASHP COP model based on temperature range
cop_slope = 0.11
cop_intercept = 3
df_model_input['cop_ashp_dynamic'] = cop_slope * df_model_input['ambient_temperature_deg_c'] + cop_intercept

# STES Parameters
capex_stes = 15      # €/kWh_th
opex_stes_frac = 0.005 # 0.5% of STES CAPEX per year
lifetime_stes = 30   # years
crf_stes = calculate_crf(discount_rate, lifetime_stes)
# The 30% annual loss represents the total recovery efficiency (70%).
annual_loss_stes = 0.30 # 30% annual self-discharge
num_timesteps = len(df_model_input)
# Calculate the fraction of energy remaining after each time step.
annual_remaining_fraction = 1 - annual_loss_stes
timestep_remaining_fraction = annual_remaining_fraction**(1 / num_timesteps)
soc_min_stes = 0.05  # Min SoC as fraction of capacity
soc_max_stes = 0.95  # Max SoC as fraction of capacity

print("All model parameters defined.")


# --- Step 4: Declaration of Gurobi Decision Variables ---
print("\n--- Step 4: Declaration of Gurobi Decision Variables ---")

# Define the time index set from the DataFrame's index
T = df_model_input.index

# --- Capacity Variables (non-negative, continuous) ---
Cap_pv = m.addVar(name="Cap_pv", lb=0)
Cap_bess_p = m.addVar(name="Cap_bess_p", lb=0)
Cap_bess_e = m.addVar(name="Cap_bess_e", lb=0)
Cap_ashp = m.addVar(name="Cap_ashp", lb=0)
Cap_stes = m.addVar(name="Cap_stes", lb=0)

# --- Operational Variables (for each time step t, non-negative, continuous) ---
P_grid_import = m.addVars(T, name="P_grid_import", lb=0)  # lb=0 prevents export
P_pv_gen = m.addVars(T, name="P_pv_gen", lb=0)
P_bess_ch = m.addVars(T, name="P_bess_ch", lb=0)
P_bess_dis = m.addVars(T, name="P_bess_dis", lb=0)
P_ashp_in = m.addVars(T, name="P_ashp_in", lb=0)
SoC_bess = m.addVars(T, name="SoC_bess", lb=0)
P_curtail = m.addVars(T, name="P_curtail", lb=0)

Q_ashp_out = m.addVars(T, name="Q_ashp_out", lb=0)
Q_stes_ch = m.addVars(T, name="Q_stes_ch", lb=0)
Q_stes_dis = m.addVars(T, name="Q_stes_dis", lb=0)
SoC_stes = m.addVars(T, name="SoC_stes", lb=0)

print("Gurobi decision variables declared.")


# --- Step 5: Formulation of the Objective Function ---
print("\n--- Step 5: Formulation of the Objective Function ---")

# Calculate total annualized CAPEX as a Gurobi linear expression
annualized_capex = (
    (crf_pv * capex_pv * Cap_pv) +
    (crf_bess * (capex_bess_p * Cap_bess_p + capex_bess_e * Cap_bess_e)) +
    (crf_ashp * capex_ashp * Cap_ashp) +
    (crf_stes * capex_stes * Cap_stes) +
    fixed_grid_cost
)

# Calculate total annual cost of grid electricity import
annual_grid_elec_cost = gp.quicksum(P_grid_import[t] * delta_t * grid_elec_price for t in T)

# Calculate total annual OPEX as a Gurobi linear expression
annual_opex = (
    (opex_pv * Cap_pv) +
    (opex_bess_frac * (capex_bess_p * Cap_bess_p + capex_bess_e * Cap_bess_e)) +
    (opex_ashp_frac * capex_ashp * Cap_ashp) +
    (opex_stes_frac * capex_stes * Cap_stes) +
    annual_grid_elec_cost
)

# Set the objective function in the model: Minimize Net Annual Cost (NAC)
m.setObjective(annualized_capex + annual_opex, GRB.MINIMIZE)
print("Objective function (Minimize Net Annual Cost) set.")


# --- Step 6: Implementation of System Constraints ---
print("\n--- Step 6: Implementation of System Constraints ---")

# Extract demand and solar data as dictionaries for faster access in the loop
D_elec = df_model_input['base_elec_demand_kw'].to_dict()
D_ev = df_model_input['scaled_ev_demand_kw'].to_dict()
D_th = df_model_input['scaled_thermal_demand_kw'].to_dict()
G_pv = df_model_input['solar_e_prod_normalized'].to_dict()
COP_ashp = df_model_input['cop_ashp_dynamic'].to_dict()


# --- Add constraints for each time step t ---
for t in T:
    # 1. Electrical Energy Balance: Generation + Discharge = Demand + Charge
    m.addConstr(P_pv_gen[t] + P_bess_dis[t] + P_grid_import[t] == D_elec[t] + D_ev[t] + P_ashp_in[t] + P_bess_ch[t], name=f"Elec_Balance_{t}")
    
    # 2. Grid Import Limit
    m.addConstr(P_grid_import[t] <= grid_import_limit_kw, name=f"Grid_Import_Limit_{t}")

    # 3. Thermal Energy Balance: Generation + Discharge = Demand + Charge
    m.addConstr(Q_ashp_out[t] + Q_stes_dis[t] == D_th[t] + Q_stes_ch[t], name=f"Therm_Balance_{t}")

    # 4. PV Generation Limit: Potential generation is split between used and curtailed power
    m.addConstr(P_pv_gen[t] + P_curtail[t] == G_pv[t] * Cap_pv * eta_pv_derate, name=f"PV_Gen_Limit_{t}")
    
    # 5. ASHP Performance and Capacity
    m.addConstr(Q_ashp_out[t] == P_ashp_in[t] * COP_ashp[t], name=f"ASHP_Perf_{t}")
    m.addConstr(Q_ashp_out[t] <= Cap_ashp, name=f"ASHP_Cap_Limit_{t}")

    # 6. BESS Power Limits
    m.addConstr(P_bess_ch[t] <= Cap_bess_p, name=f"BESS_Charge_Limit_{t}")
    m.addConstr(P_bess_dis[t] <= Cap_bess_p, name=f"BESS_Discharge_Limit_{t}")

    # 7. STES Power Limits
    m.addConstr(Q_stes_ch[t] <= Cap_ashp, name=f"STES_Charge_Limit_{t}")

    # 8. Storage SoC Balance (linking SoC between time steps)
    if t > 0:
        # BESS SoC Balance
        m.addConstr(SoC_bess[t] == (SoC_bess[t-1] * qh_remaining_fraction) + (P_bess_ch[t] * eta_bess_ch * delta_t) - ((P_bess_dis[t] / eta_bess_dis) * delta_t), name=f"BESS_SoC_Balance_{t}")
        # STES SoC Balance (including self-discharge per time step)
        m.addConstr(SoC_stes[t] == (SoC_stes[t-1] * timestep_remaining_fraction) + (Q_stes_ch[t] * delta_t) - (Q_stes_dis[t] * delta_t), name=f"STES_SoC_Balance_{t}")

    # 9. Storage SoC Limits (split into two one-sided constraints)
    # BESS SoC Limits
    m.addConstr(SoC_bess[t] >= soc_min_bess * Cap_bess_e, name=f"BESS_SoC_Min_{t}")
    m.addConstr(SoC_bess[t] <= soc_max_bess * Cap_bess_e, name=f"BESS_SoC_Max_{t}")
    # STES SoC Limits
    m.addConstr(SoC_stes[t] >= soc_min_stes * Cap_stes, name=f"STES_SoC_Min_{t}")
    m.addConstr(SoC_stes[t] <= soc_max_stes * Cap_stes, name=f"STES_SoC_Max_{t}")

# --- Initial and Final SoC Constraints ---
# These are crucial for ensuring year-over-year stability.
# Initial SoC at t=0 is set to 50% of the (unknown) capacity.
m.addConstr(SoC_bess[0] == 0.50 * Cap_bess_e, name="Initial_BESS_SoC")
m.addConstr(SoC_stes[0] == 0.50 * Cap_stes, name="Initial_STES_SoC")

# Final SoC must be >= Initial SoC to prevent energy depletion over the year.
T_final = T[-1]
m.addConstr(SoC_bess[T_final] >= SoC_bess[0], name="Final_BESS_SoC")
m.addConstr(SoC_stes[T_final] >= SoC_stes[0], name="Final_STES_SoC")

print("All system constraints implemented.")


# --- Step 7: Model Execution and Solution Extraction ---
print("\n--- Step 7: Model Execution and Solution Extraction ---")

# Display Gurobi log in console to track progress
m.Params.LogToConsole = 1 

print("\nStarting Gurobi optimization...")
m.optimize()

# --- Solution Extraction ---
results = {}
if m.Status == GRB.OPTIMAL:
    print("\nOptimal solution found!")
    
    # Extract optimal capacity values into the results dictionary
    results['PV_Capacity_kWp'] = Cap_pv.X
    results['BESS_Power_kW'] = Cap_bess_p.X
    results['BESS_Energy_kWh'] = Cap_bess_e.X
    results['ASHP_Capacity_kWth'] = Cap_ashp.X
    results['STES_Capacity_kWhth'] = Cap_stes.X
    results['Net_Annual_Cost_EUR'] = m.ObjVal

    # Extract time-series operational data for KPI calculation
    # Using .values() is a fast way to get all values from the Gurobi variable dict
    results['pv_generation_kw'] = m.getAttr('X', P_pv_gen).values()
    results['pv_curtailment_kw'] = m.getAttr('X', P_curtail).values()
    results['bess_charge_kw'] = m.getAttr('X', P_bess_ch).values()
    results['bess_discharge_kw'] = m.getAttr('X', P_bess_dis).values()
    results['ashp_input_kw'] = m.getAttr('X', P_ashp_in).values()
    results['ashp_output_kwth'] = m.getAttr('X', Q_ashp_out).values()
    results['stes_charge_kwth'] = m.getAttr('X', Q_stes_ch).values()
    results['soc_bess_kwh'] = m.getAttr('X', SoC_bess).values()
    results['soc_stes_kwhth'] = m.getAttr('X', SoC_stes).values()
    results['grid_import_kw'] = m.getAttr('X', P_grid_import).values()


else:
    print(f"\nOptimization finished with status: {m.Status}. An optimal solution was not found.")


# --- Step 8: Post-Processing, KPI Calculation, and Results Presentation ---
print("\n--- Step 8: Post-Processing, KPI Calculation, and Results Presentation ---")

if 'Net_Annual_Cost_EUR' in results:
    # --- Table 1: Optimal Capacity Configuration ---
    df_caps = pd.DataFrame({
        'Component': ['Solar PV', 'BESS Power', 'BESS Energy', 'ASHP', 'STES'],
        'Optimal Capacity': [
            results['PV_Capacity_kWp'],
            results['BESS_Power_kW'],
            results['BESS_Energy_kWh'],
            results['ASHP_Capacity_kWth'],
            results['STES_Capacity_kWhth']
        ],
        'Unit': ['kWp', 'kW', 'kWh', 'kW_th', 'kWh_th']
    })
    print("\n--- Optimal System Capacity Configuration ---")
    print(df_caps.to_string(index=False))

    # --- Table 2: Net Annual Cost (NAC) Breakdown ---
    total_variable_grid_cost = annual_grid_elec_cost.getValue()
    cost_grid = fixed_grid_cost + total_variable_grid_cost
    
    cost_pv = (crf_pv * capex_pv + opex_pv) * results['PV_Capacity_kWp']
    cost_bess = (crf_bess + opex_bess_frac) * (capex_bess_p * results['BESS_Power_kW'] + capex_bess_e * results['BESS_Energy_kWh'])
    cost_ashp = (crf_ashp * capex_ashp + opex_ashp_frac * capex_ashp) * results['ASHP_Capacity_kWth']
    cost_stes = (crf_stes * capex_stes + opex_stes_frac * capex_stes) * results['STES_Capacity_kWhth']
    
    nac_data = {
        'Component': ['PV', 'BESS', 'ASHP', 'STES', 'Grid', 'Total'],
        'Annualized CAPEX (€)': [
            crf_pv * capex_pv * results['PV_Capacity_kWp'],
            crf_bess * (capex_bess_p * results['BESS_Power_kW'] + capex_bess_e * results['BESS_Energy_kWh']),
            crf_ashp * capex_ashp * results['ASHP_Capacity_kWth'],
            crf_stes * capex_stes * results['STES_Capacity_kWhth'],
            fixed_grid_cost,
            annualized_capex.getValue()
        ],
        'Annual OPEX (€)': [
            opex_pv * results['PV_Capacity_kWp'],
            opex_bess_frac * (capex_bess_p * results['BESS_Power_kW'] + capex_bess_e * results['BESS_Energy_kWh']),
            opex_ashp_frac * capex_ashp * results['ASHP_Capacity_kWth'],
            opex_stes_frac * capex_stes * results['STES_Capacity_kWhth'],
            total_variable_grid_cost,
            annual_opex.getValue()
        ],
        'Total Annual Cost (€)': [
            cost_pv, cost_bess, cost_ashp, cost_stes, cost_grid, results['Net_Annual_Cost_EUR']
        ]
    }
    df_nac = pd.DataFrame(nac_data)
    for col in df_nac.columns[1:]:
        df_nac[col] = df_nac[col].map('{:,.0f}'.format)
    print("\n--- Net Annual Cost (NAC) Breakdown ---")
    print(df_nac.to_string(index=False))

    # --- Table 3 & 4: KPIs and System Totals ---
    # Calculate necessary annual totals
    annual_grid_import_kwh = sum(results['grid_import_kw']) * delta_t
    annual_ashp_elec_demand_kwh = sum(results['ashp_input_kw']) * delta_t
    total_system_elec_demand_kwh = annual_base_elec_kwh + annual_mobility_kwh + annual_ashp_elec_demand_kwh
    annual_bess_charge_kwh = sum(results['bess_charge_kw']) * delta_t
    annual_bess_discharge_kwh = sum(results['bess_discharge_kw']) * delta_t
    annual_stes_charge_kwh = sum(results['stes_charge_kwth']) * delta_t
    annual_pv_generation_kwh = sum(results['pv_generation_kw']) * delta_t
    annual_pv_curtailment_kwh = sum(results['pv_curtailment_kw']) * delta_t
    
    # Calculate KPIs
    ashp_capacity_factor = (sum(results['ashp_output_kwth']) * delta_t) / (results['ASHP_Capacity_kWth'] * 8760) * 100 if results['ASHP_Capacity_kWth'] > 0 else 0
    bess_efc = annual_bess_discharge_kwh / results['BESS_Energy_kWh'] if results['BESS_Energy_kWh'] > 0 else 0
    potential_pv_generation = annual_pv_generation_kwh + annual_pv_curtailment_kwh
    curtailment_rate = (annual_pv_curtailment_kwh / potential_pv_generation * 100) if potential_pv_generation > 0 else 0
    degree_of_autarky = (total_system_elec_demand_kwh - annual_grid_import_kwh) / total_system_elec_demand_kwh * 100 if total_system_elec_demand_kwh > 0 else 0

    cost_elec_system = cost_pv + cost_bess + cost_grid
    final_elec_demand = annual_base_elec_kwh + annual_mobility_kwh
    lcoe = cost_elec_system / final_elec_demand if final_elec_demand > 0 else 0

    cost_heat_system = cost_ashp + cost_stes
    cost_of_elec_for_heat = annual_ashp_elec_demand_kwh * lcoe
    lcoh = (cost_heat_system + cost_of_elec_for_heat) / annual_thermal_kwh if annual_thermal_kwh > 0 else 0

    # Segregated KPIs from System Totals
    kpi_data = {
        'KPI': [
            "Degree of Autarky (DoA)",
            "ASHP Capacity Factor",
            "BESS Equivalent Full Cycles",
            "Curtailment Rate",
            "Levelized Cost of Electricity (LCOE)",
            "Levelized Cost of Heat (LCOH)"
        ],
        'Value': [
            f"{degree_of_autarky:.2f}",
            f"{ashp_capacity_factor:.2f}",
            f"{bess_efc:.1f}",
            f"{curtailment_rate:.2f}",
            f"{lcoe:.3f}",
            f"{lcoh:.3f}"
        ],
        'Unit': ['%', '%', 'Cycles', '%', '€/kWh', '€/kWh_th']
    }
    df_kpi = pd.DataFrame(kpi_data)
    print("\n--- Key Performance Indicators ---")
    print(df_kpi.to_string(index=False))

    # Annual Demand Summary
    demand_summary_data = {
        'Demand Type': ["Annual Base Electricity Demand", "Annual Scaled Thermal Demand", "Annual Scaled Mobility Demand"],
        'Value': [f"{annual_base_elec_kwh:,.0f}", f"{annual_thermal_kwh:,.0f}", f"{annual_mobility_kwh:,.0f}"],
        'Unit': ['kWh', 'kWh,th', 'kWh']
    }
    df_demand_summary = pd.DataFrame(demand_summary_data)
    print("\n--- Annual Demand Summary ---")
    print(df_demand_summary.to_string(index=False))


    system_totals_data = {
        'Parameter': [
            "Total annual electricity demand",
            "Annual grid electricity import",
            "Annual electricity demand by ASHP",
            "Annual BESS Charge Throughput",
            "Annual STES Charge Throughput",
            "Annual PV generation",
            "Annual PV curtailment"
        ],
        'Value': [
            f"{total_system_elec_demand_kwh:,.0f}",
            f"{annual_grid_import_kwh:,.0f}",
            f"{annual_ashp_elec_demand_kwh:,.0f}",
            f"{annual_bess_charge_kwh:,.0f}",
            f"{annual_stes_charge_kwh:,.0f}",
            f"{annual_pv_generation_kwh:,.0f}",
            f"{annual_pv_curtailment_kwh:,.0f}"
        ],
        'Unit': ['kWh', 'kWh', 'kWh', 'kWh', 'kWh_th', 'kWh', 'kWh']
    }
    df_totals = pd.DataFrame(system_totals_data)
    print("\n--- Annual System Totals ---")
    print(df_totals.to_string(index=False))

    # Excel Export Section
    if openpyxl:
        try:
            excel_path = 'energy_system_optimization_results.xlsx'
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df_caps.to_excel(writer, sheet_name='Optimal_Capacities', index=False)
                # Re-create df_nac without string formatting for clean export
                nac_data_export = {
                    'Component': ['PV', 'BESS', 'ASHP', 'STES', 'Grid', 'Total'],
                    'Annualized CAPEX (€)': [
                        crf_pv * capex_pv * results['PV_Capacity_kWp'],
                        crf_bess * (capex_bess_p * results['BESS_Power_kW'] + capex_bess_e * results['BESS_Energy_kWh']),
                        crf_ashp * capex_ashp * results['ASHP_Capacity_kWth'],
                        crf_stes * capex_stes * results['STES_Capacity_kWhth'],
                        fixed_grid_cost,
                        annualized_capex.getValue()
                    ],
                    'Annual OPEX (€)': [
                        opex_pv * results['PV_Capacity_kWp'],
                        opex_bess_frac * (capex_bess_p * results['BESS_Power_kW'] + capex_bess_e * results['BESS_Energy_kWh']),
                        opex_ashp_frac * capex_ashp * results['ASHP_Capacity_kWth'],
                        opex_stes_frac * capex_stes * results['STES_Capacity_kWhth'],
                        total_variable_grid_cost,
                        annual_opex.getValue()
                    ],
                    'Total Annual Cost (€)': [cost_pv, cost_bess, cost_ashp, cost_stes, cost_grid, results['Net_Annual_Cost_EUR']]
                }
                df_nac_export = pd.DataFrame(nac_data_export)
                df_nac_export.to_excel(writer, sheet_name='NAC_Breakdown', index=False)
                df_kpi.to_excel(writer, sheet_name='Key_Performance_Indicators', index=False)
                df_demand_summary.to_excel(writer, sheet_name='Annual_Demand_Summary', index=False)
                df_totals.to_excel(writer, sheet_name='Annual_System_Totals', index=False)
            print(f"\nSuccessfully exported results to '{excel_path}'")
        except Exception as e:
            print(f"\nAn error occurred while exporting to Excel: {e}")

    # Plot BESS State of Charge
    soc_bess_pct = []
    if results['BESS_Energy_kWh'] > 0:
        soc_bess_kwh_list = list(results['soc_bess_kwh'])
        soc_bess_pct = [(val / results['BESS_Energy_kWh']) * 100 for val in soc_bess_kwh_list]
    plt.figure(figsize=(15, 7))
    plt.plot(df_model_input.index, soc_bess_pct, label='BESS SoC', color='teal')
    plt.ylabel('State of Charge (%)')
    plt.xlabel('Time Step (Quarter Hour of Year)')
    plt.title('BESS State of Charge Over the Year')
    plt.grid(True)
    plt.ylim(0, 100)
    plt.legend()
    plt.show()

    # Plot STES State of Charge
    soc_stes_pct = []
    if results['STES_Capacity_kWhth'] > 0:
        soc_stes_kwhth_list = list(results['soc_stes_kwhth'])
        soc_stes_pct = [(val / results['STES_Capacity_kWhth']) * 100 for val in soc_stes_kwhth_list]
    plt.figure(figsize=(15, 7))
    plt.plot(df_model_input.index, soc_stes_pct, label='STES SoC', color='firebrick')
    plt.ylabel('State of Charge (%)')
    plt.xlabel('Time Step (Quarter Hour of Year)')
    plt.title('STES State of Charge Over the Year')
    plt.grid(True)
    plt.ylim(0, 100)
    plt.legend()
    plt.show()

    # Calculate total consumption for each time step
    total_consumption_kw = df_model_input['base_elec_demand_kw'] + df_model_input['scaled_ev_demand_kw'] + list(results['ashp_input_kw'])
    # Plot Total Electricity Consumption
    plt.figure(figsize=(15, 7))
    plt.plot(df_model_input.index, total_consumption_kw, label='Total Electricity Consumption', color='darkviolet')
    plt.ylabel('Power (kW)')
    plt.xlabel('Time Step (Quarter Hour of Year)')
    plt.title('Total Electricity Consumption Profile Over the Year')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot Grid Electricity Import
    plt.figure(figsize=(15, 7))
    plt.plot(df_model_input.index, results['grid_import_kw'], label='Grid Import', color='saddlebrown')
    plt.ylabel('Power (kW)')
    plt.xlabel('Time Step (Quarter Hour of Year)')
    plt.title('Grid Electricity Import Profile Over the Year')
    plt.grid(True)
    plt.legend()
    plt.show()

else:
    print("\nCould not generate results as no optimal solution was found.")