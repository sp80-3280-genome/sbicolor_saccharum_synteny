import definition_synteny_r570_ as ds
import sys
import pandas as pd
simulation = pd.DataFrame(columns=['contig', 'organism', 'markers', 'bmax', 'bcount', 'c_size','simulation'])
(ft,og) = ds.load_tables('ftable.2.tsv', 'sacch_r570.cds.blast.e-5.og.tsv')
for x in range(int(sys.argv[1]), int(sys.argv[2])):
    z = ds.random_distribution(ft,og)
    a = ds.fake_ft(ft,z,organism='Saccharum_R570')
    a_ = a[(a.organism =='Sobic')|(a.organism =='SP803280')]
    b = ds.make_df(a_.drop_duplicates(),og)
    c = ds.df_final(b,a_)
    a_ = a[(a.organism =='Sobic')|(a.organism =='Saccharum_R570')]
    b = ds.make_df(a_.drop_duplicates(),og)
    c = c.append(ds.df_final(b,a_))
    c['simulation'] = x
    simulation = simulation.append(c)
simulation.to_csv('simulation_novo{}'.format(sys.argv[2]), sep='\t', index=False)
