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

# === Property Details === 
# MSSubClass, MSZoning, Neighborhood, HouseStyle, OverallQual, OverallCond, Functional

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏠 Property Details", "📅 Home Age", "🚗 Garage & Basement", "🔥 Interior & Features", "📏 Additional Space", "💰 Sale Details"])
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
    
    with col1:
        house_styles_values = list(house_styles_mapping.keys())
        house_styles_labels = list(house_styles_mapping.values())
        selected_style = st.selectbox("Select House Style", house_styles_labels)
        house_style = house_styles_values[house_styles_labels.index(selected_style)]
        
    with col2:                                                  
        # Filter valid house styles based on selected MSSubClass
        allowed_subclass_values = house_style_to_subclass.get(house_style, [])
        allowed_subclass_labels = [ms_subclass_mapping[val] for val in allowed_subclass_values]

        selected_subclass = st.selectbox("Select Property Type", allowed_subclass_labels)
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
        selected_MSZoning = st.selectbox("Select Zoning", mszoning_labels)
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
            "Names": "North Ames",
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
        selected_neighborhood = st.selectbox("Select Neighborhood", neighborhood_labels)
        Neighborhood = neighborhood_values[neighborhood_labels.index(selected_neighborhood)]
    
    
    st.divider()
    st.write("Living Area")
    col7, col8, col9 = st.columns(3)
    #will be more specific for these features later
    # TotalSF(Engineered) = 1stFlrSF, 2ndFlrSF, BsmtFinSF1, BsmFinSF2
    # Total Area(Engineered) = GrLivArea(Any Area that is above ground), TotalBsmtSF
    # TotalBathrooms(Engineered) = BsmtFullBath, FullBath, 0.5 * HalfBath, BsmtHalfBath
    with col7:
        GrLivArea = st.number_input("Above Grade Living Area (sq ft)", 100, 10000, 1500) 
    with col8:
        FirstFlrSF = st.number_input("First Floor Area (sq ft)", 100, 5000, 1150)
    with col9:
        SecondFlrSF = st.number_input("Second Floor Area (sq ft)", 0, 3000, 350)

    col10, col11 = st.columns(2)
    with col10:
        FullBath = st.number_input("Number of Full Bathrooms", 0, 3, 1, step=1) 
    with col11:    
        HalfBath = st.number_input("Number of Half Bathrooms", 0, 2, 1, step=1)
    #Functional - Home Functionality Rating
    
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
        selected_exterqual = st.selectbox("Exterior Material Quality", exterqual_labels)
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
        selected_functional = st.selectbox("Home Functionality Rating", functional_labels)
        Functional = functional_values[functional_labels.index(selected_functional)]
    
    st.divider()
    

# === Home Age === 
# HouseAge = YrSold - YearBuilt
# HouseRemodelAge = YrSold - YearRemodAdd
with tab2:
    #get current year
    current_year = datetime.datetime.now().year

    #Streamlit input for year built and calculate HouseAge
    yearBuilt = st.number_input("Year Built", 1800, current_year, 2025)
    HouseAge = current_year - yearBuilt
    st.write(f"House Age: {HouseAge} years")
    #Streamlit input for year remodeling age and calculate for HouseRemodelAge
    yearRemodAdd = st.number_input("Year of Remodeling", 1800, current_year, 2025)
    HouseRemodelAge = current_year - yearRemodAdd
    st.write(f"House Remodeling Age: {HouseRemodelAge} years")
    
        #SaleCondition = st.selectbox("Sale Condition", ["Normal", "Abnorml", "Partial", "AdjLand", "Alloca"])

        #PoolArea = st.number_input("Pool Area (sq ft)", 0, 1000, 0)
        #HasPool = st.radio("Has Pool?", [0, 1], index=0)

# === Garage & Basement ===
# GarageFinish, GarageCars, BsmtQual, BsmtExposure, BsmtFinType1, BsmtUnfSF
with tab3:
    st.subheader("Basement", divider="grey")
    bscol1, bscol2, bscol3 = st.columns(3)
    with bscol1:
        #Total Basement Area
        TotalBsmtSF = st.number_input("Total Basement Area", 0, 7500, 1000)
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
        selected_bsmtqual = st.selectbox("Height of the Basement", bsmtqual_labels)
        BsmtQual = bsmtqual_values[bsmtqual_labels.index(selected_bsmtqual)]

    with bscol2:
        BsmtFinSF = st.number_input("Basement Finished Area", 0, 7000, 500)
        BsmtUnfSF = st.number_input("Basement Unfinished Area", 0, 2500, 500)

    with bscol3:
        bsmtfintype_mapping = {
            "GLQ": "Good Living Quarters",
            "ALQ": "Average Living Quarters",
            "BLQ": "Below Average Living Quarters",
            "Rec": "Average Rec Room",
            "LwQ": "Low Quality",
            "Unf": "Unfinshed",
            "NA": "No Basement",
        }
    
        bsmtfintype_values = list(bsmtfintype_mapping.keys())
        bsmtfintype_labels = list(bsmtfintype_mapping.values())
        selected_bsmtfintype = st.selectbox("Rating of Finished Basement Area", bsmtfintype_labels)
        BsmtFinType1 = bsmtfintype_values[bsmtfintype_labels.index(selected_bsmtfintype)]

        bsmtexposure_mapping = {
            "Gd": "Good Exposure",
            "Av": "Average Exposure",
            "Mn": "Mimimum Exposure",
            "No": "No Exposure",
            "NA": "No Basement",
        }
    
        bsmtexposure_values = list(bsmtexposure_mapping.keys())
        bsmtexposure_labels = list(bsmtexposure_mapping.values())
        selected_bsmtexposure = st.selectbox("Rating of Walkout/Garden Level Walls", bsmtexposure_labels)
        BsmtExposure = bsmtexposure_values[bsmtexposure_labels.index(selected_bsmtexposure)]

    bscol4, bscol5 = st.columns(2)
    with bscol4:
        BsmtFullBath = st.number_input("Full Bathrooms in Basement", 0, 3, 1, step=1)
    with bscol5:
        BsmtHalfBath = st.number_input("Half Bathrooms in Basement", 0, 2, 1, step=1)

    st.subheader("Garage", divider="grey")

    garagefinish_mapping = {
            "Fin": "Finished",
            "RFn": "Rough Finished",
            "Unf": "Unfinished",
            "NA": "No Garage",
    }
    
    garagefinish_values = list(garagefinish_mapping.keys())
    garagefinish_labels = list(garagefinish_mapping.values())
    selected_garagefinish = st.selectbox("Garage's Finish", garagefinish_labels)
    GarageFinish = garagefinish_values[garagefinish_labels.index(selected_garagefinish)]

    GarageCars = st.slider("Garage Car Capacity", 0, 5, 2)

# === Interior & Features ===
# KitchenAbvGr, KitchenQual, Fireplaces, FireplaceQu, BedroomAbvGr, TotRmsAbvGrd, 
# MasVnrType, MasVnrArea, ExterQual, Heating, HeatingQc, CentralAir
with tab4:
    st.subheader("Rooms", divider="grey")
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        # BedroomAbvGr - Number of bedrooms above basement level
        BedroomAbvGr = st.number_input("Number of Bedrooms Above Basement Level", min_value=0, max_value=8, value=3, step=1)
    with rcol2:
        # TotRmsAbvGrd - Total rooms above grade
        TotRmsAbvGrd = st.number_input("Total Rooms Above Grade", min_value=2, max_value=14, value=6, step=1)
    
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
        selected_kitchenqual = st.selectbox("Kitchen Quality", kitchenqual_labels)
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
        selected_heating = st.selectbox("Heating", heating_labels)
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
        selected_heatingqc = st.selectbox("Heating Quality", heatingqc_labels)
        HeatingQC = heatingqc_values[heatingqc_labels.index(selected_heatingqc)]

    with hcol3:
        air_option = st.radio("Central Air Conditioning?", ["Yes", "No"], index=1)
        CentralAir = 1 if air_option == "Yes" else 0

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        Fireplaces = st.slider("Number of Fireplaces", 0, 3, 0)
    with fcol2:
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
        selected_fireplacequ = st.selectbox("Fireplace Quality", fireplacequ_labels)
        FireplaceQu = fireplacequ_values[fireplacequ_labels.index(selected_fireplacequ)]

    st.subheader("Masonry Veneer Type", divider="red")
    mvcol1, mvcol2 = st.columns(2)
    with mvcol1:
        masvnrtype_mapping = {
            "BrkCmn": "Brick Common",
            "BrkFace": "Brick Face",
            "CBlock": "Cinder Block",
            "None": "None",
            "Stone": "Stone"
        }
        masvnrtype_values = list(masvnrtype_mapping.keys())
        masvnrtype_labels = list(masvnrtype_mapping.values())
        selected_masvnrtype = st.selectbox("Masonry Veneer Type", masvnrtype_labels)
        MasVnrType = masvnrtype_values[masvnrtype_labels.index(selected_masvnrtype)]

    with mvcol2:
        # MasVnrArea - Masonry veneer area in square feet
        MasVnrArea = st.slider("Masonry Veneer Area (sq ft)", min_value=0, max_value=1500, value=200, step=10)



    

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
        pool_option = st.radio("Has Pool?", ["Yes", "No"], index=1)
        HasPool = 1 if pool_option == "Yes" else 0
       

    with pcol2:
        PoolArea = st.slider("Pool Area", 0, 750, 0)

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

    # Calculate TotalPorchSF Automatically
    

    

# === Sale & Transaction Details
# MoSold, SaleType, SaleCondition,


# Create user input dictionary and ensure all expected features exist
#initialize with all expected features set to zero
TotalSF = FirstFlrSF + SecondFlrSF + BsmtFinSF
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
st.success(f"💰 The estimated house price is **${predicted_price[0]:,.2f}**")


# === House Visualization (Simple Representation) ===
#with st.expander("📏 House Layout Visualization", expanded=True):
    