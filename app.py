import streamlit as st
import pickle
import numpy as np
import pandas as pd
import datetime

# Load the trained stacking regressor model from the pickle file
with open('stacking_model.pkl', 'rb') as f:
    stack_model = pickle.load(f)

with open('preprocessing_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)  # Load the saved preprocessing pipeline
# Define the feature names that the model expects (update this list to match your training features)

with open('expected_features.pkl', 'rb') as f:
    expected_features = pickle.load(f)  # Load the expected feature names

# Streamlit UI
st.title("🏡 Real Estate House Price Prediction in Ames, Iowa")
st.write("Enter house details to estimate the price.")

# Create a placeholder for the predicted price at the top of the app
predicted_price_placeholder = st.empty()

# === Property Details === 
# MSSubClass, MSZoning, Neighborhood, HouseStyle, OverallQual, OverallCond, Functional

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Property Details", "📅 Home Age", "🚗 Garage & Basement", "🔥 Interior & Features", "📏 Additional Space"])
with tab1:
    st.write("Property Type & Location")
    col1, col2 = st.columns(2)
    house_styles_mapping = {
        "1Story": "1-Story Home",
        "1.5Fin": "1.5-Story: 2nd Level Finished",
        "1.5Unf": "1.5-Story: 2nd Level Unfinished",
        "2Story": "2-Story Home",
        "2.5Fin": "2.5-Story: 2nd Level Finished",
        "2.5Unf": "2.5-Story: 2nd Level Unfinished",
        "SFoyer": "Split Foyer",
        "SLvl": "Split Level"
    }
    # Mapping of MSSubClass to valid HouseStyles
    # Allowed MSSubClass per HouseStyle
    house_style_to_subclass = {
        "1Story": [20, 30, 40, 90, 120, 190],
        "1.5Fin": [30, 50, 90, 190],
        "1.5Unf": [30, 45, 190],
        "2Story": [20, 50, 60, 70, 75, 90, 160, 190],
        "2.5Fin": [70, 75, 190],
        "2.5Unf": [75, 190],
        "SFoyer": [85, 90, 120, 180],
        "SLvl": [20, 60, 80, 90, 180, 190]
    }
    
    ms_subclass_mapping = {
        20: "1-Story (Modern)",
        30: "1-Story (Old, Pre-1946)",
        40: "1-Story + Attic",
        45: "1.5-Story (Unfinished)",
        50: "1.5-Story (Finished)",
        60: "2-Story (Modern)",
        70: "2-Story (Old, Pre-1946)",
        75: "2.5-Story",
        80: "Split-Level",
        85: "Split Foyer",
        90: "Duplex",
        120: "1-Story PUD",
        150: "1.5-Story PUD",
        160: "2-Story PUD",
        180: "Multi-Level PUD",
        190: "2-Family Conversion",
    }

    house_style_to_default_subclass = {
        "1Story": 20,  # 1-Story (Modern)
        "1.5Fin": 50,  # 1.5-Story (Finished)
        "1.5Unf": 45,  # 1.5-Story (Unfinished)
        "2Story": 60,  # 2-Story (Modern)
        "2.5Fin": 75,  # 2.5-Story
        "2.5Unf": 75,  # 2.5-Story
        "SFoyer": 85,  # Split Foyer
        "SLvl": 80,    # Split-Level
    }
    
    with col1:
        house_styles_values = list(house_styles_mapping.keys())
        house_styles_labels = list(house_styles_mapping.values())
        house_styles_default_index = house_styles_labels.index("1-Story Home")
        selected_style = st.selectbox("Select House Style", house_styles_labels, index=house_styles_default_index)
        house_style = house_styles_values[house_styles_labels.index(selected_style)]
        
    with col2:                                                  
        # Filter valid house styles based on selected MSSubClass
        allowed_subclass_values = house_style_to_subclass.get(house_style, [])
        allowed_subclass_labels = [ms_subclass_mapping[val] for val in allowed_subclass_values]

        # Get the default MSSubClass for the selected HouseStyle
        default_subclass = house_style_to_default_subclass.get(house_style, allowed_subclass_values[0])
        default_subclass_label = ms_subclass_mapping[default_subclass]
        default_subclass_index = allowed_subclass_labels.index(default_subclass_label)

        selected_subclass = st.selectbox("Select Property Type", allowed_subclass_labels, index=default_subclass_index)
        ms_subclass = allowed_subclass_values[allowed_subclass_labels.index(selected_subclass)]

    col3, col4 = st.columns(2)    
    with col3:
        mszoning_mapping = {
            "A": "Agriculture",
            "C": "Commercial",
            "FV": "Floating Village Residential",
            "I": "Industrial",
            "RH": "Residential High Density",
            "RL": "Residential Low Density",
            "RP": "Residential Low Density Park",
            "RM": "Residential Medium Density"
        }
        mszoning_values = list(mszoning_mapping.keys())
        mszoning_labels = list(mszoning_mapping.values())
        mszoning_default_index =  mszoning_labels.index("Residential Low Density")
        selected_MSZoning = st.selectbox("Select Zoning", mszoning_labels, index=mszoning_default_index)
        MSZoning = mszoning_values[mszoning_labels.index(selected_MSZoning)]
    with col4:
        neighborhood_mapping = {
            "Blmngtn":"Bloomington Heights",
            "Blueste": "Bluestem",
            "BrDale": "Briardale",
            "BrkSide": "Brookside",
            "ClearCr": "Clear Creek",
            "CollgCr": "College Creek",
            "Crawfor": "Crawford",
            "Edwards": "Edwards",
            "Gilbert": "Gilbert",
            "IDOTRR": "Iowa DOT and Rail Road",
            "MeadowV": "Meadow Village",
            "Mitchel": "Mitchell",
            "NAmes": "North Ames",
            "NoRidge": "Northridge",
            "NPkVill": "Northpark Villa",
            "NridgHt": "Northridge Heights",
            "NWAmes": "Northwest Ames",
            "OldTown": "Old Town",
            "SWISU": "South & West of Iowa State University",
            "Sawyer": "Sawyer",
            "SawyerW": "Sawyer West",
            "Somerst": "Somerset",
            "StoneBr": "Stone Brook",
            "Timber": "Timberland",
            "Veenker": "Veenker"
        }
        neighborhood_values = list(neighborhood_mapping.keys())
        neighborhood_labels = list(neighborhood_mapping.values())
        neighborhood_default_index = neighborhood_labels.index("North Ames")
        selected_neighborhood = st.selectbox("Select Neighborhood", neighborhood_labels, index=neighborhood_default_index)
        Neighborhood = neighborhood_values[neighborhood_labels.index(selected_neighborhood)]
    
    
    st.divider()
    st.write("Living Area")
    
    col7, col8, col9 = st.columns(3)
    #will be more specific for these features later
    # TotalSF(Engineered) = 1stFlrSF, 2ndFlrSF, BsmtFinSF1, BsmFinSF2
    # Total Area(Engineered) = GrLivArea(Any Area that is above ground), TotalBsmtSF
    # TotalBathrooms(Engineered) = BsmtFullBath, FullBath, 0.5 * HalfBath, BsmtHalfBath
    with col7:
        GrLivArea = st.number_input("Above Ground Living Area (sq ft)", 100, 10000, 1500) 

    
    with col8:
        if house_style == "1Story":
            # For 1-story houses, FirstFlrSF must equal GrLivArea
            FirstFlrSF = st.number_input("First Floor Area (sq ft)", 100, 10000, GrLivArea, disabled=True)
        else:
            # For multi-story houses, allow the user to input FirstFlrSF
            FirstFlrSF = st.number_input("First Floor Area (sq ft)", 100, GrLivArea - 100, 800)
    with col9:
        if house_style == "1Story":
            SecondFlrSF = st.number_input("Second Floor Area (sq ft)", 0, 3000, 0, disabled=True)
        else:
            SecondFlrSF = st.number_input("Second Floor Area (sq ft)", 0, GrLivArea-FirstFlrSF, GrLivArea - FirstFlrSF, disabled=True)
            


    col10, col11 = st.columns(2)
    with col10:
        FullBath = st.slider("Number of Full Bathrooms", 0, 3, 1, step=1) 
    with col11:    
        HalfBath = st.slider("Number of Half Bathrooms", 0, 2, 1, step=1)
    
    st.divider()
    st.write("Quality & Condition")
    col5, col6 = st.columns(2)  
    with col5:
        overall_quality_mapping = {
            10: "Very Excellent",
            9: "Excellent",
            8: "Very Good",
            7: "Good",
            6: "Above Average",
            5: "Average",
            4: "Below Average",
            3: "Fair",
            2: "Poor",
            1: "Very Poor",
        }
        # Overall Quality Slider with description
        OverallQual = st.slider(
            "Overall Quality (Material & Finish of Home)",
            min_value=1,
            max_value=10,
            value=5,  # Default value
            step=1
        )
        st.write(f"Selected Quality: {OverallQual} - {overall_quality_mapping[OverallQual]}")
    
    with col6:
        overall_cond_mapping = {
            10: "Very Excellent",
            9: "Excellent",
            8: "Very Good",
            7: "Good",
            6: "Above Average",
            5: "Average",
            4: "Below Average",
            3: "Fair",
            2: "Poor",
            1: "Very Poor",
        }
        OverallCond = st.slider(
            "Overall Condition",
            min_value=1,
            max_value=10,
            value=5,  # Default value
            step=1
        )
        st.write(f"Selected Condition: {OverallCond} - {overall_cond_mapping[OverallCond]}")
    
    col7, col8 = st.columns(2)
    with col7:
        # ExterQual - Exterior Quality
        exterqual_mapping = {
            "Ex": "Excellent",
            "Gd": "Good",
            "TA": "Average/Typical",
            "Fa": "Fair",
            "Po": "Poor"
        }
        exterqual_values = list(exterqual_mapping.keys())
        exterqual_labels = list(exterqual_mapping.values())
        exterqual_default_index = exterqual_labels.index("Average/Typical")
        selected_exterqual = st.selectbox("Exterior Material Quality", exterqual_labels, index=exterqual_default_index)
        ExterQual = exterqual_values[exterqual_labels.index(selected_exterqual)]
    with col8:
        functional_mapping = {
            "Typ": "Typical Functionality",
            "Min1": "Minor Deductions 1",
            "Min2": "Minor Deductions 2",
            "Mod": "Moderate Deductions",
            "Maj1": "Major Deductions 1",
            "Maj2": "Major Deductions 2",
            "Sev": "Severely Damaged",
            "Sal": "Salvage only"
        }
        functional_values = list(functional_mapping.keys())
        functional_labels = list(functional_mapping.values())
        functional_default_index = functional_labels.index("Typical Functionality")
        selected_functional = st.selectbox("Home Functionality Rating", functional_labels, index=functional_default_index)
        Functional = functional_values[functional_labels.index(selected_functional)]
    
    st.divider()
    

# === Home Age === 
# HouseAge = YrSold - YearBuilt
# HouseRemodelAge = YrSold - YearRemodAdd
with tab2:
    #get current year
    current_year = datetime.datetime.now().year

    #Streamlit input for year built and calculate HouseAge
    yearBuilt = st.number_input("Year Built", 1800, current_year, 1973)
    HouseAge = current_year - yearBuilt
    st.write(f"House Age: {HouseAge} years")
    #Streamlit input for year remodeling age and calculate for HouseRemodelAge
    yearRemodAdd = st.number_input("Year of Remodeling", 1800, current_year, 1994)
    HouseRemodelAge = current_year - yearRemodAdd
    st.write(f"House Remodeling Age: {HouseRemodelAge} years")

    st.divider()

# === Garage & Basement ===
# GarageFinish, GarageCars, BsmtQual, BsmtExposure, BsmtFinType1, BsmtUnfSF
with tab3:
    st.subheader("Basement", divider="grey")
    bscol1, bscol2, bscol3 = st.columns(3)
    with bscol1:
        #Total Basement Area
        TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", 0, 7500, 1000)
        #Height of the Basement
        bsmtqual_mapping = {
            "Ex": "Excellent (100+ inches)",
            "Gd": "Good (90-99 inches)",
            "TA": "Typical (80-89 inches)",
            "Fa": "Fair (70-79 inches)",
            "Po": "Poor (<70 inches)",
            "NA": "No Basement",
        }
        bsmtqual_values = list(bsmtqual_mapping.keys())
        bsmtqual_labels = list(bsmtqual_mapping.values())
        #default
        # Filter options based on TotalBsmtSF
        if TotalBsmtSF == 0:
            # Only show "No Basement" if TotalBsmtSF is 0
            bsmtqual_options = {"NA": "No Basement"}
        else:
            # Exclude "No Basement" if TotalBsmtSF is greater than 0
            bsmtqual_options = {k: v for k, v in bsmtqual_mapping.items() if k != "NA"}

        # Convert filtered options to lists
        bsmtqual_values = list(bsmtqual_options.keys())
        bsmtqual_labels = list(bsmtqual_options.values())
        # If TotalBsmtSF is 0, set default to "No Basement"
        if TotalBsmtSF == 0:
            bsmtqual_default_index = bsmtqual_labels.index("No Basement")
        else:
            bsmtqual_default_index = bsmtqual_labels.index("Typical (80-89 inches)")

       

        selected_bsmtqual = st.selectbox("Height of the Basement", bsmtqual_labels, index=bsmtqual_default_index)
        BsmtQual = bsmtqual_values[bsmtqual_labels.index(selected_bsmtqual)]

        # Display a warning if the selection is invalid
        if TotalBsmtSF == 0 and BsmtQual != "NA":
            st.warning("You cannot select a basement height when there is no basement. Please select 'No Basement'.")
        elif TotalBsmtSF > 0 and BsmtQual == "NA":
            st.warning("You cannot select 'No Basement' when there is a basement. Please choose a valid basement height.")


    with bscol3:
        BsmtFinSF = st.number_input("Basement Finished Area (sq ft)", 0, TotalBsmtSF, min(1000, TotalBsmtSF))
        BsmtUnfSF = st.number_input("Basement Unfinished Area (sq ft)", 0, TotalBsmtSF-BsmtFinSF, TotalBsmtSF-BsmtFinSF) 

    with bscol2:
        bsmtfintype_mapping = {
            "GLQ": "Good Living Quarters",
            "ALQ": "Average Living Quarters",
            "BLQ": "Below Average Living Quarters",
            "Rec": "Average Rec Room",
            "LwQ": "Low Quality",
            "Unf": "Unfinished",
            "NA": "No Basement",
        }

        if TotalBsmtSF == 0:
            # Only show "No Basement" if TotalBsmtSF is 0
            bsmtfintype_options = {"NA": "No Basement"}
        else:
            # Exclude "No Basement" if TotalBsmtSF is greater than 0
            bsmtfintype_options = {k: v for k, v in bsmtfintype_mapping.items() if k != "NA"}
    
        bsmtfintype_values = list(bsmtfintype_options.keys())
        bsmtfintype_labels = list(bsmtfintype_options.values())
        #Default value
        # If TotalBsmtSF, BsmtFinSF, or BsmtUnfSF is 0, set default to "No Basement"
        if TotalBsmtSF == 0:
            bsmtfintype_default_index = bsmtfintype_labels.index("No Basement")
        else:
            bsmtfintype_default_index = bsmtfintype_labels.index("Good Living Quarters")
        
        selected_bsmtfintype = st.selectbox("Rating of Finished Basement Area", bsmtfintype_labels, index=bsmtfintype_default_index)
        BsmtFinType1 = bsmtfintype_values[bsmtfintype_labels.index(selected_bsmtfintype)]

        bsmtexposure_mapping = {
            "Gd": "Good Exposure",
            "Av": "Average Exposure",
            "Mn": "Mimimum Exposure",
            "No": "No Exposure",
            "NA": "No Basement",
        }

        if TotalBsmtSF == 0:
            # Only show "No Basement" if TotalBsmtSF is 0
            bsmtexposure_options = {"NA": "No Basement"}
        else:
            # Exclude "No Basement" if TotalBsmtSF is greater than 0
            bsmtexposure_options = {k: v for k, v in bsmtexposure_mapping.items() if k != "NA"}
    
        bsmtexposure_values = list(bsmtexposure_options.keys())
        bsmtexposure_labels = list(bsmtexposure_options.values())

        # If TotalBsmtSF, BsmtFinSF, or BsmtUnfSF is 0, set default to "No Basement"
        if TotalBsmtSF == 0:
            bsmtexposure_default_index = bsmtexposure_labels.index("No Basement")
        else:
            bsmtexposure_default_index = bsmtexposure_labels.index("No Exposure")

        selected_bsmtexposure = st.selectbox("Rating of Walkout/Garden Level Walls", bsmtexposure_labels, index=bsmtexposure_default_index)
        BsmtExposure = bsmtexposure_values[bsmtexposure_labels.index(selected_bsmtexposure)]

    bscol4, bscol5 = st.columns(2)
    with bscol4:
        if TotalBsmtSF == 0:
            BsmtFullBath = st.number_input("Full Bathrooms in Basement", 0, 0, 0, step=0)
        else:
            BsmtFullBath = st.number_input("Full Bathrooms in Basement", 0, 3, 1, step=1)
    with bscol5:
        if TotalBsmtSF == 0:
            BsmtHalfBath = st.number_input("Half Bathrooms in Basement", 0, 0, 0, step=0)
        else:
            BsmtHalfBath = st.number_input("Half Bathrooms in Basement", 0, 2, 0, step=1)

    if TotalBsmtSF == 0:
        st.write("To get more options, please enter a value for Total Basement Area.")

    st.subheader("Garage", divider="grey")

    garagefinish_mapping = {
        "Fin": "Finished",
        "RFn": "Rough Finished",
        "Unf": "Unfinished",
        "NA": "No Garage",
    }

    
    garagefinish_values = list(garagefinish_mapping.keys())
    garagefinish_labels = list(garagefinish_mapping.values())

    #Default value
    garagefinish_default_index = garagefinish_labels.index("Unfinished")

    #display
    selected_garagefinish = st.selectbox("Garage's Finish", garagefinish_labels, index=garagefinish_default_index)
    GarageFinish = garagefinish_values[garagefinish_labels.index(selected_garagefinish)]
    if GarageFinish == "NA":
        GarageCars = st.slider("Garage Car Capacity", 0, 5, 0, disabled=True)
    else:
        GarageCars = st.slider("Garage Car Capacity", 0, 5, 2)

    st.divider()

# === Interior & Features ===
# KitchenAbvGr, KitchenQual, Fireplaces, FireplaceQu, BedroomAbvGr, TotRmsAbvGrd, 
# MasVnrType, MasVnrArea, ExterQual, Heating, HeatingQc, CentralAir
with tab4:
    st.subheader("Rooms", divider="grey")
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        # BedroomAbvGr - Number of bedrooms above basement level
        BedroomAbvGr = st.slider("Number of Bedrooms Above Basement Level", min_value=0, max_value=8, value=2, step=1)
    with rcol2:
        # TotRmsAbvGrd - Total rooms above grade
        TotRmsAbvGrd = st.slider("Total Rooms Above Ground", min_value=2, max_value=14, value=6, step=1)
    
    st.subheader("Kitchen", divider="grey")
    #Kitchen Above Grade
    #Kitchen Quality
    kcol1, kcol2 = st.columns(2)
    
    with kcol1:
        KitchenAbvGr = st.slider("Number of Kitchen", 0, 3, 1)
    
    with kcol2:
        kitchenqual_mapping = {
                "Ex": "Excellent",
                "Gd": "Good",
                "TA": "Typical/Average",
                "Fa": "Fair",
                "Po": "Poor",
        }
    
        kitchenqual_values = list(kitchenqual_mapping.keys())
        kitchenqual_labels = list(kitchenqual_mapping.values())
        kitchenqual_default_index = kitchenqual_labels.index("Typical/Average")
        selected_kitchenqual = st.selectbox("Kitchen Quality", kitchenqual_labels, index=kitchenqual_default_index)
        KitchenQual = kitchenqual_values[kitchenqual_labels.index(selected_kitchenqual)]
    
    st.subheader("Heating/Air", divider="grey")

    hcol1, hcol2, hcol3 = st.columns(3)

    with hcol1:
        heating_mapping = {
            "Floor": "Floor Furnace",
            "GasA": "Gas Forced Warm Air Furnace",
            "GasW": "Gas Hot Water or Steam Heat",
            "Grav": "Gravity Furnace",
            "OthW": "Hot Water or Stream Heat Other Than Gas",
            "Wall": "Wall Furnace",
        }
    
        heating_values = list(heating_mapping.keys())
        heating_labels = list(heating_mapping.values())
        heating_default_index = heating_labels.index("Gas Forced Warm Air Furnace")
        selected_heating = st.selectbox("Heating", heating_labels, index=heating_default_index)
        Heating = heating_values[heating_labels.index(selected_heating)]
    
    with hcol2:
        heatingqc_mapping = {
            "Ex": "Excellent",
            "Gd": "Good",
            "TA": "Typical/Average",
            "Fa": "Fair",
            "Po": "Poor",
        }

        heatingqc_values = list(heatingqc_mapping.keys())
        heatingqc_labels = list(heatingqc_mapping.values())
        heatingqc_default_index = heatingqc_labels.index("Typical/Average")
        selected_heatingqc = st.selectbox("Heating Quality", heatingqc_labels, index=heatingqc_default_index)
        HeatingQC = heatingqc_values[heatingqc_labels.index(selected_heatingqc)]

    with hcol3:
        air_option = st.radio("Central Air Conditioning?", ["Yes", "No"], index=1)
        CentralAir = 1 if air_option == "Yes" else 0

    st.subheader('Fireplace', divider="red")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        fireplacequ_mapping = {
            "Ex": "Excellent - Exceptional Masonry Fireplace",
            "Gd": "Good - Masonry Fireplace in main level",
            "TA": "Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement",
            "Fa": "Fair - Prefabricated Fireplace in basement",
            "Po": "Poor - Ben Franklin Stove",
            "NA": "No Fireplace",
        }

        fireplacequ_values = list(fireplacequ_mapping.keys())
        fireplacequ_labels = list(fireplacequ_mapping.values())
        fireplacequ_default_index = fireplacequ_labels.index("No Fireplace")
        selected_fireplacequ = st.selectbox("Fireplace Quality", fireplacequ_labels, index=fireplacequ_default_index)
        FireplaceQu = fireplacequ_values[fireplacequ_labels.index(selected_fireplacequ)]

    with fcol2:
        if FireplaceQu == 'NA':
            Fireplaces = st.slider("Number of Fireplaces", 0, 3, 0, disabled=True)
        else:
            Fireplaces = st.slider("Number of Fireplaces", 0, 3, 1)
        

    st.subheader("Masonry Veneer Type", divider="red")
    mvcol1, mvcol2 = st.columns(2)
    with mvcol1:
        masvnrtype_mapping = {
            "BrkCmn": "Brick Common",
            "BrkFace": "Brick Face",
            "CBlock": "Cinder Block",
            "Stone": "Stone",
            "None": "None",
        }
        masvnrtype_values = list(masvnrtype_mapping.keys())
        masvnrtype_labels = list(masvnrtype_mapping.values())
        masvnrtype_default_index = masvnrtype_labels.index("None")
        selected_masvnrtype = st.selectbox("Masonry Veneer Type", masvnrtype_labels, index=masvnrtype_default_index)
        MasVnrType = masvnrtype_values[masvnrtype_labels.index(selected_masvnrtype)]

    with mvcol2:
        # MasVnrArea - Masonry veneer area in square feet
        if MasVnrType == "None":
            MasVnrArea = st.slider("Masonry Veneer Area (sq ft)", min_value=0, max_value=1500, value=0, step=10, disabled=True)
        else:
            MasVnrArea = st.slider("Masonry Veneer Area (sq ft)", min_value=0, max_value=1500, value=180, step=10)

    st.divider()



    

# === Additional Space & Lot Size ===
# LotFrontage, LotArea, TotalPorchSF(Engineered), PoolArea, HasPool
with tab5: 
    st.subheader("Lot", divider="grey")
    lotcol1, lotcol2 = st.columns(2)
    with lotcol1:
        LotFrontage = st.slider("Lot Frontage", 0, 200, 70)
    with lotcol2:
        LotArea = st.slider("Lot Area", 0, 55000, 10500)

    st.subheader("Pool", divider="blue")

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        pool_option = st.radio("Is there a Pool?", ["Yes", "No"], index=1)
        HasPool = 1 if pool_option == "Yes" else 0
       

    with pcol2:
         # Pool Area slider
        if pool_option == "Yes":
            PoolArea = st.slider("Pool Area (sq ft)", 0, 750, 500)
        else:
            # Disable the slider but keep it visible
            PoolArea = st.slider("Pool Area (sq ft)", 0, 750, 0, disabled=True)

    st.subheader("Porch", divider="grey")
    # Individual Inputs for Each Porch Type
    porchcol1, porchcol2 = st.columns(2)
    with porchcol1:
        WoodDeckSF = st.number_input("Wood Deck Area (sq ft)", min_value=0, max_value=1000, value=0, step=10)
    with porchcol2:
        OpenPorchSF = st.number_input("Open Porch Area (sq ft)", min_value=0, max_value=1000, value=0, step=10)
    
    porchcol3, porchcol4 = st.columns(2)
    with porchcol3:  
        EnclosedPorch = st.number_input("Enclosed Porch Area (sq ft)", min_value=0, max_value=1000, value=0, step=10)
    with porchcol4:
        ThreeSsnPorch = st.number_input("Three Season Porch Area (sq ft)", min_value=0, max_value=1000, value=0, step=10)
 
    ScreenPorch = st.number_input("Screen Porch Area (sq ft)", min_value=0, max_value=1000, value=0, step=10)

    st.divider()

    # Calculate TotalPorchSF Automatically
    

    

# === Sale & Transaction Details
# MoSold, SaleType, SaleCondition,


# Create user input dictionary and ensure all expected features exist
#initialize with all expected features set to zero
TotalSF = FirstFlrSF + SecondFlrSF + BsmtFinSF + BsmtUnfSF
TotalBathrooms = BsmtFullBath + FullBath + 0.5 * (HalfBath + BsmtHalfBath)
TotalArea = GrLivArea + TotalBsmtSF
TotalPorchSF = WoodDeckSF + OpenPorchSF + EnclosedPorch + ThreeSsnPorch + ScreenPorch

user_input = {feature: 0 for feature in expected_features}

#update with user-selected values
user_input.update({
    "OverallQual": OverallQual,
    "OverallCond": OverallCond,
    "TotalArea": TotalArea,
    "TotalSF": TotalSF,
    "HouseAge": HouseAge,
    "HouseRemodelAge": HouseRemodelAge,
    "Fireplaces": Fireplaces,
    "FireplaceQu": FireplaceQu,
    "GarageCars": GarageCars,
    "PoolArea": PoolArea,
    "HasPool": HasPool,
    "MSZoning": MSZoning,
    "Neighborhood": Neighborhood,
    #"SaleCondition": SaleCondition,
    'BsmtUnfSF' : BsmtUnfSF,
    "MSSubClass": ms_subclass,
    "LotFrontage": LotFrontage,
    "LotArea": LotArea,
    "HouseStyle": house_style,
    "BsmtQual": BsmtQual,
    "BsmtExposure": BsmtExposure,
    "TotalBathrooms": TotalBathrooms,
    "BsmtFinType1": BsmtFinType1,
    "GarageFinish": GarageFinish,
    "KitchenAbvGr": KitchenAbvGr,
    "KitchenQual": KitchenQual,
    "Heating": Heating,
    "HeatingQC": HeatingQC,
    "CentralAir": CentralAir,
    "Functional": Functional,
    "TotRmsAbvGrd": TotRmsAbvGrd,
    "BedroomAbvGr": BedroomAbvGr,
    "MasVnrType": MasVnrType,
    "MasVnrArea": MasVnrArea,
    "ExterQual": ExterQual,
    "TotalPorchSF": TotalPorchSF,

})

# Convert user input to DataFrame
user_input_df = pd.DataFrame([user_input])
# Apply preprocessing to handle categorical encoding
processed_features = pipeline.transform(user_input_df)
st.write(user_input_df)
#st.write(processed_features)
# === Live Price Prediction ===
predicted_price = stack_model.predict(processed_features)
predicted_price_placeholder.success(f"💰 The estimated house price is **${predicted_price[0]:,.2f}**")


# === House Visualization (Simple Representation) ===
#with st.expander("📏 House Layout Visualization", expanded=True):
    