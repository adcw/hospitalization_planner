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
    C.SEPSIS_MYCOTICA_CULTURE,                    # ['YES', 'NO']
    C.UREAPLASMA_CULTURE,                        # ['YES', 'NO']
    C.UREAPLASMA,                               # ['No_Ureoplazma0' 'No_Ureoplazma1' 'No_Ureoplazma2' 'No_Ureoplazma3',
                                                # 'Ureoplazma0' 'Ureoplazma1' 'Ureoplazma2' 'Ureoplazma3']

    C.RDS,                                      # ['No_RDS0' 'No_RDS1' 'No_RDS2' 'No_RDS3' 'RDS0' 'RDS1' 'RDS2' 'RDS3']
    C.RTG_RDS,                                  # ['YES', 'NO']
    C.RDS_TYPE,                                 # ['1' '2' '3' '4' 'NO']
    C.GENERAL_SURFRACTANT,                       # ['YES', 'NO']
    C.PDA,                                      # ['No_PDA0' 'No_PDA1' 'No_PDA2' 'No_PDA3' 'PDA0' 'PDA1' 'PDA2' 'PDA3']
    C.RTG_PDA,                                  # ['YES', 'NO']
    C.PDA_CLOSED,                               # ['YES', 'NO']
    C.RESPCODE,                                 # ['HFO' 'IMV' 'WLASNY' 'n-CPAP' 't-CPAP']

    #### OUTPUT DATA                            # ['YES', 'NO']
    # Drugs
    C.DOPAMINE, C.DOBUTAMINE, C.LEVONOR, C.ADRENALINE,
    C.IMV, C.HFO, C.CPAP, C.SURFRACTANT,
    C.MAP1, C.MAP2, C.MAP3,
    C.PENICILINE, C.PENICILINE_2,
    C.MACROLIDE, C.CEPHALOSPORIN, C.AMINOGLYCOSIDE, C.CARBAPENEM, C.CEPHALOSPORINE34, C.GLYCOPEPTIDE
]
