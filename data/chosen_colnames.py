import data.colnames_original as c

"""
['FiO2', 'ANTYBIOTYK', 'RTG_RDS', 'AMINA_PRESYJNA', 'dopamina',
       'po2', 'dobutamina', 'AMINOGLIKOZYD', 'ptl', 'STERYD',
       'birthweight', 'RTG_PDA', 'GENERAL_PDA_CLOSED', 'adrenalina',
       'PENICELINA1', 'GENERAL_SURFACTANT', 'KARBAPENEM']
"""

COLS = [
    c.PATIENTID,
    c.DATEID,
    c.BIRTHWEIGHT,

    c.FIO2,
    c.PO2,
    c.PTL,
    # c.CREATININE,
    # c.TOTAL_BILIRUBIN,

    # c.POSIEW_SEPSA,
    # c.POSIEW_SEPSA_MYCOTICA,
    # c.POSIEW_UREOPLAZMA,
    # c.POSIEW_UREOPLAZMA,
    c.RTG_RDS,
    # c.TYPE_RDS,
    c.RTG_PDA,

    c.GENERAL_SURFACTANT,
    # c.SURFACTANT,
    # c.PDA_CLOSED,
    c.GENERAL_PDA_CLOSED,
    # c.CPAP,
    # c.MAP1,
    # c.MAP2,
    # c.MAP3,

    # c.DOBUTAMINA,
    # c.LEVONOR,
    # c.DOPAMINA, # ???
    c.ADRENALINA,
    c.PENICELINA1,
    # c.PENICELINA2,
    # c.CEFALOSPORYNA2,
    # c.CEFALOSPORYNA34,
    c.KARBAPENEM,
    # c.MAKROLIT,
    c.AMINOGLIKOZYD,
    # c.GLIKOPEPTYD,
    c.AMINA_PRESYJNA,
    # c.HEMOSTATYCZNY,
    c.STERYD,
    # c.P_GRZYBICZNY,
    # c.GAMAGLOBULIN,
    # c.FLUOROCHINOLON,
    c.ANTYBIOTYK,
    c.RESPIRATION
]
