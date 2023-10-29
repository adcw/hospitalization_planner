import data.colnames as C

colnames = [
    #### INPUT DATA
    # IDs
    C.PATIENT_ID, C.DATE_ID,                      # [<int>]

    # Constant attributes
    C.BIRTHWEIGHT,                              # [<float>]

    # Continuous attributes
    C.PTL, C.CREATININE, C.TOTAL_BILIRUBIN,     # [<float>, 'MISSING'] - the only columns that has missing values
                                                # Missing value percentage
                                                # C.PTL                 0.094 %
                                                # C.CREATININE          22.03 %
                                                # C.TOTAL_BILIRUBIN     0.164 %

    C.FIO2, C.PO2,                              # [<float>]

    # Discrete attributes
    C.SEPSIS_CULTURE,                             # ['NO' 'Sepsa_minus' 'Sepsa_mycotica' 'Sepsa_plus' 'Sepsa_unknown']
    C.UREAPLASMA_CULTURE,                        # ['YES', 'NO']

    C.RDS_TYPE,                                 # ['1' '2' '3' '4' 'NO']
    C.GENERAL_SURFRACTANT,                       # ['YES', 'NO']
    C.RTG_PDA,                                  # ['YES', 'NO']
    C.PDA_CLOSED,                               # ['YES', 'NO']
    C.RESPCODE,                                 # ['HFO' 'IMV' 'WLASNY' 'n-CPAP' 't-CPAP']

    #### OUTPUT DATA                            # ['YES', 'NO']
    # Drugs
    C.DOPAMINE, C.DOBUTAMINE, C.LEVONOR, C.ADRENALINE, C.SURFRACTANT,
    C.MAP1, C.MAP2, C.MAP3,
    C.PENICILINE, C.PENICILINE_2,
    C.MACROLIDE, C.CEPHALOSPORIN, C.AMINOGLYCOSIDE, C.CARBAPENEM, C.CEPHALOSPORINE34, C.GLYCOPEPTIDE
]
