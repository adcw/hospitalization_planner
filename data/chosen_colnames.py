import data.colnames as C

colnames = [
    #### INPUT DATA
    # IDs
    C.PATIENTID, C.DATEID,                      # [<int>]

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
    C.POSIEW_SEPSA,                             # ['NO' 'Sepsa_minus' 'Sepsa_mycotica' 'Sepsa_plus' 'Sepsa_unknown']
    C.POSIEW_SEPSA_MYCOTICA,                    # ['YES', 'NO']
    C.POSIEW_UREOPLAZMA,                        # ['YES', 'NO']
    C.UREOPLAZMA,                               # ['No_Ureoplazma0' 'No_Ureoplazma1' 'No_Ureoplazma2' 'No_Ureoplazma3',
                                                # 'Ureoplazma0' 'Ureoplazma1' 'Ureoplazma2' 'Ureoplazma3']

    C.RDS,                                      # ['No_RDS0' 'No_RDS1' 'No_RDS2' 'No_RDS3' 'RDS0' 'RDS1' 'RDS2' 'RDS3']
    C.RTG_RDS,                                  # ['YES', 'NO']
    C.TYPE_RDS,                                 # ['1' '2' '3' '4' 'NO']
    C.GENERAL_SURFACTANT,                       # ['YES', 'NO']
    C.PDA,                                      # ['No_PDA0' 'No_PDA1' 'No_PDA2' 'No_PDA3' 'PDA0' 'PDA1' 'PDA2' 'PDA3']
    C.RTG_PDA,                                  # ['YES', 'NO']
    C.PDA_CLOSED,                               # ['YES', 'NO']
    C.RESPCODE,                                 # ['HFO' 'IMV' 'WLASNY' 'n-CPAP' 't-CPAP']

    #### OUTPUT DATA                            # ['YES', 'NO']
    # Drugs
    C.DOPAMINA, C.DOBUTAMINA, C.LEVONOR, C.ADRENALINA,
    C.IMV, C.HFO, C.CPAP, C.SURFACTANT,
    C.MAP1, C.MAP2, C.MAP3,
    C.PENICYLINA1, C.PENICYLINA2,
    C.MAKROLIT, C.CEFALOSPORYNA2, C.AMINOGLIKOZYD, C.KARBAPENEM, C.CEFALOSPORYNA34, C.GLIKOPEPTYD
]
