# V9.2_a direct-year 2D-field MVEOF peak audit summary

version: `v9_2_a`
output_tag: `direct_year_2d_meof_peak_audit_v9_2_a`

## Method boundary
- Samples are real years 1979-2023, not bootstrap-resampled year combinations.
- X is built from 2D object fields, not zonal/profile features.
- MVEOF modes are unsupervised field modes; they are not target-guided peak-order modes.
- Peak timing is computed on PC high/mid/low group composites, not on single years.
- PC high/low groups are score phases of real years, not named physical regimes.

## Configuration
- target_windows: W045, W081, W113, W160
- spatial_res_deg: 2.0
- n_modes_main: 3; n_modes_save: 5

## Mode overview
- W045 mode 1: EVR=0.0856
- W045 mode 2: EVR=0.0728
- W045 mode 3: EVR=0.0690
- W045 mode 4: EVR=0.0523
- W045 mode 5: EVR=0.0464
- W081 mode 1: EVR=0.0879
- W081 mode 2: EVR=0.0798
- W081 mode 3: EVR=0.0598
- W081 mode 4: EVR=0.0544
- W081 mode 5: EVR=0.0537
- W113 mode 1: EVR=0.0845
- W113 mode 2: EVR=0.0682
- W113 mode 3: EVR=0.0620
- W113 mode 4: EVR=0.0475
- W113 mode 5: EVR=0.0470
- W160 mode 1: EVR=0.0854
- W160 mode 2: EVR=0.0697
- W160 mode 3: EVR=0.0572
- W160 mode 4: EVR=0.0510
- W160 mode 5: EVR=0.0474

## PC high/low real years
- W045 mode 1 high: 1994, 2003, 1980, 1995, 2010, 2019, 2015, 2021, 1987, 2023, 1983, 1991, 2018, 2016, 1998
- W045 mode 1 low: 1984, 2000, 1996, 2009, 1992, 1982, 1997, 2008, 1989, 1985, 1981, 1979, 2007, 1999, 2013
- W045 mode 2 high: 1990, 2005, 2004, 1981, 2011, 1988, 1982, 2020, 2003, 2002, 1993, 1985, 1992, 1998, 1997
- W045 mode 2 low: 1991, 1999, 1995, 2000, 2013, 2014, 2008, 1996, 2019, 2023, 1986, 1984, 1994, 1989, 2012
- W045 mode 3 high: 1987, 2000, 2006, 2013, 2017, 1982, 1996, 1980, 1989, 1988, 1991, 2015, 2011, 2021, 2020
- W045 mode 3 low: 2003, 1986, 1995, 2002, 1983, 1979, 2012, 1981, 2010, 2009, 1999, 2023, 2008, 1992, 2019
- W081 mode 1 high: 1996, 1999, 1988, 2011, 1987, 2015, 1980, 2023, 1998, 2020, 2018, 2013, 2016, 1991, 2021
- W081 mode 1 low: 1992, 1982, 1997, 1979, 1985, 2002, 1986, 2003, 1993, 2012, 2004, 1983, 2008, 2007, 1984
- W081 mode 2 high: 2010, 2016, 1980, 2023, 1993, 2021, 1987, 1995, 1991, 2014, 1982, 1992, 2020, 2019, 1983
- W081 mode 2 low: 1984, 1981, 2001, 2000, 2018, 1994, 2011, 1985, 2013, 2008, 2022, 2012, 2006, 1990, 2009
- W081 mode 3 high: 1981, 2001, 2003, 1994, 1993, 1990, 1996, 2016, 1979, 2022, 1997, 2010, 2017, 2020, 1998
- W081 mode 3 low: 1987, 1982, 2004, 2014, 1991, 1985, 2018, 1995, 1986, 2013, 2015, 2009, 1983, 1999, 2000
- W113 mode 1 high: 2023, 2009, 1980, 2014, 2010, 2021, 1983, 1995, 2017, 2022, 1993, 1987, 2003, 1998, 2020
- W113 mode 1 low: 2000, 2018, 1984, 1994, 1986, 2001, 1985, 1981, 1999, 2012, 2002, 2004, 2011, 1989, 1997
- W113 mode 2 high: 2011, 2006, 2005, 2012, 1998, 2023, 2016, 1981, 2017, 2013, 2021, 2010, 2018, 1994, 2022
- W113 mode 2 low: 1983, 2015, 1982, 1980, 1992, 1987, 1986, 1993, 1979, 2002, 2019, 1999, 2014, 2004, 1991
- W113 mode 3 high: 2005, 1989, 1987, 1981, 1998, 1988, 2000, 2017, 1993, 2015, 1994, 2014, 1999, 1980, 2021
- W113 mode 3 low: 2022, 2016, 1983, 1979, 2007, 1992, 2013, 2020, 2006, 1986, 2019, 1985, 1982, 2010, 1984
- W160 mode 1 high: 1995, 2005, 1980, 2023, 1987, 2009, 1993, 2014, 2020, 2003, 2022, 2010, 2017, 2021, 1998
- W160 mode 1 low: 1986, 1984, 2000, 1989, 1981, 1985, 1990, 1982, 2012, 2016, 1997, 2006, 1992, 1979, 1991
- W160 mode 2 high: 1981, 1988, 1991, 1996, 2000, 1997, 2002, 2009, 1995, 1982, 1993, 2015, 2014, 1987, 1980
- W160 mode 2 low: 2022, 2010, 2007, 2016, 2013, 2023, 2018, 2006, 1985, 2012, 2020, 2005, 2019, 2003, 1979
- W160 mode 3 high: 1998, 1993, 2023, 1980, 2016, 2004, 2015, 2005, 2007, 1983, 1999, 2001, 1989, 2008, 1988
- W160 mode 3 low: 1995, 1997, 1994, 2002, 1990, 2020, 2009, 2022, 2011, 2003, 2012, 1996, 1992, 2019, 1987

## Peak-relevance overview
- W045 mode 1: peak_relevant_moderate; order_change_pairs=3
- W045 mode 2: peak_relevant_strong; order_change_pairs=6
- W045 mode 3: peak_relevant_strong; order_change_pairs=6
- W081 mode 1: peak_relevant_strong; order_change_pairs=6
- W081 mode 2: peak_relevant_strong; order_change_pairs=7
- W081 mode 3: peak_relevant_strong; order_change_pairs=8
- W113 mode 1: peak_relevant_moderate; order_change_pairs=7
- W113 mode 2: peak_relevant_strong; order_change_pairs=5
- W113 mode 3: peak_relevant_moderate; order_change_pairs=7
- W160 mode 1: peak_relevant_strong; order_change_pairs=9
- W160 mode 2: peak_relevant_strong; order_change_pairs=7
- W160 mode 3: peak_relevant_strong; order_change_pairs=7

## Forbidden interpretations
- Do not interpret PC high/low as named physical year types without later physical audit.
- Do not interpret MVEOF maximum-variance modes as peak-order mechanisms by default.
- Do not interpret group-composite peak timing as a single-year rule.
- Do not infer causality from high/low order contrasts.