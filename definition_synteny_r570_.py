#!/usr/bin/env python3
"""
Load python3 from env
"""
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

pd.set_option('chained_assignment', None)

def load_tables(feature_table_file, union_finder_file):
    """
    Load Feature table and union finder files as pandas DataFrame.
    """
    union_finder = pd.read_csv(union_finder_file, sep="\t")
    feature_table = pd.read_csv(feature_table_file, sep="\t")

    # Load tables
    union_finder.columns = ['locus_tag', 'gid', 'gsize']
    union_finder['org'] = union_finder['locus_tag'].apply(
        lambda x: np.where((x[0:5] != 'Sspon') & (x[0:5] != 'Sobic') &
                           (x[0:2] != 'Sh'), 'SP803280', x[0:5]))
    union_finder['org'] = np.where(union_finder.org.str.startswith('Sh'),'R570',
                                  union_finder.org)
    union_finder.sort_values(['gid', 'org', 'locus_tag'], inplace=True)

    # Build df from
    feature_table = feature_table.merge(
        union_finder[['locus_tag', 'gid']], left_on='locus_tag', right_on='locus_tag', how='left')
    feature_table['oidx'] = np.where(feature_table.organism == 'Sobic', 0,
                                     np.where(feature_table.organism ==
                                              'Sspon',
                                              1,np.where(feature_table.organism
                                                        == 'SP803280',2, 3)))
    return (feature_table, union_finder)

def make_df(feature_table, union_finder, maximun_distance=10,
            reference_organism=False):
    """
    Take an ftable and unionfinder dataframe as input and returns a dataframe
    to perform synteny analysis
    """
    # Detec how many genomes are being compared:
    genomes_compared = len(set(union_finder.org))
    # Count number of loci per orthologous group
    orthologous_group = union_finder.groupby(
        ['gid', 'org']).agg({'gsize':'first', 'locus_tag':'count'}
                           ).reset_index()
    tmp = orthologous_group.pivot(
        index='gid', columns='org', values='locus_tag').reset_index()
    orthologous_group = tmp.merge(
        orthologous_group[['gid', 'gsize']].drop_duplicates(),
        left_on='gid', right_on='gid', how='left'
    )
    tmp = None
    orthologous_group['norg'] = genomes_compared - (orthologous_group.isnull().sum(axis=1))

    # Find OGs of Sorghum single copy genes and are presented in all genomes
    # being compared
    gid = set(orthologous_group[(orthologous_group.norg == genomes_compared) &
                                (orthologous_group.Sobic == 1)].gid)

    synteny_df = feature_table[feature_table.gid.isin(gid)].merge(
        feature_table[feature_table.gid.isin(gid)], left_on='gid', right_on='gid', how='inner')
    synteny_df = synteny_df[synteny_df.oidx_x < synteny_df.oidx_y]
    synteny_df.sort_values(
        ['oidx_x', 'genomic_accession_x', 'oidx_y', 'genomic_accession_y',
         'genomic_order_x', 'genomic_order_y'], inplace=True
    )

    gene_per_contig = feature_table[feature_table.gid.isin(gid)].groupby(
        ['organism', 'genomic_accession']).agg({'locus_tag':'count'}).reset_index()
    gene_per_contig_1 = set(
        gene_per_contig[gene_per_contig.locus_tag == 1].genomic_accession.values)
    synteny_df = synteny_df[~(synteny_df.genomic_accession_x.isin(gene_per_contig_1) |
                              synteny_df.genomic_accession_y.isin(gene_per_contig_1))]
    synteny_df = synteny_df[
        ['organism_x', 'genomic_accession_x', 'gstart_x', 'gend_x', 'strand_x',
         'genomic_order_x', 'organism_y', 'genomic_accession_y', 'gstart_y', 'gend_y',
         'strand_y', 'genomic_order_y', 'gid', 'locus_tag_x', 'locus_tag_y']
    ]
    synteny_df.columns = [
        'org1', 'ga1', 'gs1', 'ge1', 's1', 'go1', 'org2', 'ga2', 'gs2',
        'ge2', 's2', 'go2', 'gid', 'lt1', 'lt2'
    ]
    if reference_organism:
        synteny_df = synteny_df[synteny_df.org1 == reference_organism]
    synteny_df.sort_values(
        ['org1', 'ga1', 'org2', 'ga2', 'go2', 'go1'], inplace=True                # swap gos
    )

    # Count loci per contig
    contig_count = feature_table.groupby(['genomic_accession']).agg({'ftable_id':'count'})
    contig_count.reset_index(inplace=True)
    synteny_df = synteny_df.merge(
        contig_count, left_on=['ga1'], right_on=['genomic_accession'], how='left'
    )
    synteny_df.drop(columns='genomic_accession', inplace=True)
    synteny_df = synteny_df.merge(
        contig_count, left_on=['ga2'], right_on=['genomic_accession'], how='left'
    )
    synteny_df.drop(columns='genomic_accession', inplace=True)
    # nlt1 = total number of loci in the first contig of this row
    # nlt2 = total number of loci in the second contig of this row
    synteny_df.columns = [
        'org1', 'ga1', 'gs1', 'ge1', 's1', 'go1', 'org2', 'ga2', 'gs2', 'ge2',
        's2', 'go2', 'gid', 'lt1', 'lt2', 'nlt1', 'nlt2']

    # Count number of locus per contig in orthologous groups
    contig_count = synteny_df[['ga1', 'lt1']].drop_duplicates()
    contig_count_2 = synteny_df[['ga2', 'lt2']].drop_duplicates()
    contig_count.columns = ['ga', 'nt']
    contig_count_2.columns = ['ga', 'nt']
    contig_count = contig_count.append(contig_count_2).drop_duplicates()
    contig_count = contig_count.groupby('ga').agg({'nt':'nunique'})
    contig_count.reset_index(inplace=True)

    # Add columns with the number of loci found in df
    synteny_df = synteny_df.merge(contig_count, left_on=['ga1'], right_on=['ga'], how='left')
    synteny_df.drop(columns='ga', inplace=True)
    synteny_df = synteny_df.merge(contig_count, left_on=['ga2'], right_on=['ga'], how='left')
    synteny_df.drop(columns='ga', inplace=True)
    # nt1 = number of loci in df found for the first contig of this row
    # nt2 = number of loci in df found for the second contig of this row
    synteny_df.columns = [
        'org1', 'ga1', 'gs1', 'ge1', 's1', 'go1', 'org2', 'ga2', 'gs2', 'ge2', 's2', 'go2',
        'gid', 'lt1', 'lt2', 'nlt1', 'nlt2', 'nt1', 'nt2'
    ]

    # Find blocks
    synteny_df['t1'] = np.where((synteny_df.ga1 == synteny_df.ga1.shift(1)) &
                                (synteny_df.ga2 == synteny_df.ga2.shift(1)),
                                (synteny_df.go1 - synteny_df.go1.shift(1)),
                                np.nan)
    synteny_df['t2'] = np.where((synteny_df.ga1 == synteny_df.ga1.shift(1)) &
                                (synteny_df.ga2 == synteny_df.ga2.shift(1)),
                                (synteny_df.go2 - synteny_df.go2.shift(1)),
                                np.nan)

    synteny_df['criteria'] = \
                    np.where(synteny_df.t1.isna(), 'contig changed',
                    np.where(synteny_df.t2.isna(), 'contig changed',
                    np.where((abs(abs(synteny_df.t1) - abs(synteny_df.t2)) >= maximun_distance), 'unequal marker distance',
                    np.where(((synteny_df.s1 == synteny_df.s2) != (synteny_df.s1.shift(1) == synteny_df.s2.shift(1))), 'asymmetric strand reversal',
                    np.where(((abs(abs(synteny_df.t1.shift(1)) - abs(synteny_df.t2.shift(1))) < maximun_distance)
                    & ((synteny_df.t1 * synteny_df.t1.shift(1) < 0) | (synteny_df.t2 * synteny_df.t2.shift(1) < 0))), 'translocation or duplication',
                    'colinear')))))

    synteny_df['block'] = \
    (
        (synteny_df.t1.isna() | synteny_df.t2.isna() |
         (abs(abs(synteny_df.t1) - abs(synteny_df.t2)) >= maximun_distance)
        ) | (
            (synteny_df.s1 == synteny_df.s2) != (synteny_df.s1.shift(1) == synteny_df.s2.shift(1))
            )| (
                (abs(abs(synteny_df.t1.shift(1)) - abs(synteny_df.t2.shift(1))) < maximun_distance)
                & (synteny_df.criteria.shift(1) != 'translocation or duplication')
                & ((synteny_df.t1 * synteny_df.t1.shift(1) < 0)
                   | (synteny_df.t2 * synteny_df.t2.shift(1) < 0)))
    ).cumsum()

    # Fix problem with translocation of or across multiple genes
    synteny_df.loc[(synteny_df.criteria == 'translocation or duplication') & (synteny_df.block == synteny_df.block.shift(1)),'criteria'] = 'colinear'

    return synteny_df


def random_distribution(feature_table,uf, size=10000, organism='SP803280',
                        ft_filter=False):
    """
    Get n numbers of contigs (size) from a feature_table DataFrame from a
    specific organism and create a dataframe with the distribution of genes
    per contig
    """
    if ft_filter:
        orthologous_group = uf_stats(uf)
        gid = set(orthologous_group[(orthologous_group.norg == 3) &
                                  (orthologous_group.Sobic == 1)].gid)
        feature_table = feature_table[feature_table.gid.isin(gid)]
    random_ga = random.sample(
        set(feature_table[feature_table['organism'] == organism].genomic_accession
           ), int(size))
    distribution = feature_table[feature_table['genomic_accession'].isin(random_ga)].groupby(
        'genomic_accession').agg({'locus_tag':'count'}
                                ).locus_tag.value_counts().to_frame().reset_index()
    distribution.columns = ['contig_size', 'number_of_contigs']
    distribution.sort_values('contig_size', ascending=False, inplace=True)
    return distribution, random_ga

def fake_ft(feature_table, distribution, organism='Sspon'):
    """
    Create a fake feature table containing the genes from a given distribution
    """
    organism_to_fake = feature_table[feature_table.organism == organism].reset_index(drop=True)
    fake = pd.DataFrame(columns=organism_to_fake.columns)
    fake_count = 0
    idx = 0
    idx_already_taken = []
    for number, times in zip(list(distribution[0].contig_size),
                             list(distribution[0].number_of_contigs)):
        time = 0
        while time != times:
            idx = random.sample(set(organism_to_fake.index), 1)[0]
            idx_list = list(range(idx, idx+number))
            set_contig = list(set(organism_to_fake.iloc[idx:idx + number].genomic_accession.values))
            if (len(set_contig) == 1) & (
                    idx + number+ 1 < len(organism_to_fake)
            ) & (bool(set(idx_already_taken).intersection(idx_list)) is False):
                fake_count += 1
                time += 1
                idx_already_taken.extend(idx_list)
                fake_contig = organism_to_fake.iloc[idx_list]
                fake_contig.genomic_order = range(len(fake_contig))
                #fake_contig.loc[:, ['genomic_accession']] = 'fake_contig_{}'.format(fake_count)
                fake_contig['genomic_accession'] = 'fake_contig_{}'.format(fake_count)
                fake = fake.append(fake_contig)
            else:
                pass
    fake_ft_result = feature_table[feature_table.genomic_accession.isin(distribution[1])]
    fake_ft_result = fake_ft_result.append(feature_table[feature_table.organism == 'Sobic'])
    fake_ft_result = fake_ft_result.append(fake)
    return fake_ft_result

def df_to_distribution(synteny_df, reference_organism='Sobic'):
    """
    Take a synteny df results as arument and create a distribution with
    percent using a model organism to compare.
    """
    block_count = synteny_df.groupby('block').agg(
        {'org1':'first', 'org2':'first', 'lt1':'nunique', 'lt2':'nunique'})
    block_count.reset_index(inplace=True)
    block_count.columns = ['block', 'org1', 'org2', 'unique_lt1', 'unique_lt2']
    block_count['count_orthologous_lt'] = np.where(
        block_count.unique_lt1 <= block_count.unique_lt2,
        block_count.unique_lt1, block_count.unique_lt2)
    block_count = block_count.groupby(
        ['org1', 'org2', 'count_orthologous_lt']).agg({'block': 'count'}).reset_index()
    block_count.columns = ['org1', 'org2', 'nlt', 'nblock']
    group = block_count.groupby(['org1', 'org2'])
    bc_with_percent = pd.DataFrame(columns=['org1', 'org2', 'nlt', 'nblock'])
    for _, block_count_dfs  in group:
        block_count_dfs = block_count_dfs[(block_count_dfs.org1 == reference_organism)]
        block_count_dfs['blocks_percent'] = block_count_dfs.nblock/block_count_dfs.nblock.sum()
        block_count_dfs['total_genes'] = block_count_dfs.nblock * block_count_dfs.nlt
        block_count_dfs['genes_percent'] = \
                block_count_dfs.total_genes/ block_count_dfs.total_genes.sum()
        bc_with_percent = bc_with_percent.append(block_count_dfs)
    bc_with_percent.total_genes = bc_with_percent.total_genes.astype(int)
    return bc_with_percent


def plot_df_distribution(distribution_df, frequency_type='a', frequency_count='blocks'):
    """ take a distribuition plot as argument and plot the distribuition with
    bar plots, frequency_type can be showed as relative (r),absolute(a) or
    both(b)
    """
    if frequency_count == 'blocks':
        distribution_df = distribution_df[['nblock', 'nlt', 'org1', 'org2', 'blocks_percent']]
        distribution_df.columns = ['nblock', 'Block size', 'org1', 'Species', 'percent']
        distribution_plot = distribution_df.pivot(columns='Species', values='nblock',
                                                  index='Block size')
        distribution_plot_re = distribution_df.pivot(columns='Species', values='percent',
                                                     index='Block size')
    elif frequency_count == 'genes':
        distribution_df = distribution_df[['total_genes', 'nlt', 'org1',
                                           'org2', 'genes_percent']]
        distribution_df.columns = ['Total of genes', 'Block size', 'org1', 'Species', 'percent']
        distribution_plot = distribution_df.pivot(columns='Species', values='Total of genes',
                                                  index='Block size')
        distribution_plot_re = distribution_df.pivot(columns='Species', values='percent',
                                                     index='Block size')
    if (frequency_type == 'a') & (frequency_count == 'blocks'):
        result = distribution_plot
        plot = result.plot.bar(rot=0)
        plot.set(ylabel="Number of blocks")
    elif (frequency_type == 'r') & (frequency_count == 'blocks'):
        result = distribution_plot_re
        plot = result.plot.bar(rot=0)
        plot.set(ylabel="Relative number of blocks")
    elif (frequency_type == 'log') & (frequency_count == 'blocks'):
        result = distribution_plot
        plot = result.plot.bar(rot=0)
        plot.set_yscale('log')
        plot.set(ylabel="Number of blocks")
    elif (frequency_type == 'a') & (frequency_count == 'genes'):
        result = distribution_plot
        plot = result.plot.bar(rot=0)
        plot.set(ylabel="Number of genes")
    elif (frequency_type == 'r') & (frequency_count == 'genes'):
        result = distribution_plot_re
        plot = result.plot.bar(rot=0)
        plot.set(ylabel="Relative number of genes")
    elif (frequency_type == 'log') & (frequency_count == 'genes'):
        result = distribution_plot
        plot = result.plot.bar(rot=0)
        plot.set_yscale('log')
        plot.set(ylabel="Number of genes")

    plot.set(xlabel="Block size")

    return plot

def plot_ft_distribution(distribution_ft, frequency_type='a', frequency_count='contigs'):
    """
    Escrever doc string
    """

    distribution_ft['bin'] = pd.cut(
        distribution_ft['contig_size'], [0, 1, 2, 10, 100, 1000, 10000000],
        labels=[1, 2, '3-10', '11-100', '101-1000', '>1000'])

    if frequency_count == 'contigs':
        distribution_ft = distribution_ft.groupby(
            ['organism', 'bin']).agg(
                {'genes_percent':'sum', 'number of genes':'sum',
                 'number_of_contigs':'sum', 'contig_percent':'sum'}).reset_index()
        distribution_plot = distribution_ft.pivot(
            columns='organism', values='number_of_contigs', index='bin')
        distribution_plot_re = distribution_ft.pivot(
            columns='organism', values='contig_percent', index='bin')
    if frequency_count == 'genes':
        distribution_ft = distribution_ft.groupby(
            ['organism', 'bin']).agg(
                {'genes_percent':'sum', 'number of genes':'sum',
                 'number_of_contigs':'sum', 'contig_percent':'sum'}).reset_index()
        distribution_plot = distribution_ft.pivot(
            columns='organism', values='number of genes', index='bin')
        distribution_plot_re = distribution_ft.pivot(
            columns='organism', values='genes_percent', index='bin')

    if (frequency_type == 'a') & (frequency_count == 'contigs'):
        result = distribution_plot
        plot = result.plot.bar(rot=0)
        plot.set(ylabel="Number of contigs")
    elif (frequency_type == 'r') & (frequency_count == 'contigs'):
        result = distribution_plot_re
        plot = result.plot.bar(rot=0)
        plot.set(ylabel="Relative number of contigs")
    elif (frequency_type == 'log') & (frequency_count == 'contigs'):
        result = distribution_plot
        plot = result.plot.bar(rot=0)
        plot.set_yscale('log')
        plot.set(ylabel="Number of contigs")
    elif (frequency_type == 'a') & (frequency_count == 'genes'):
        result = distribution_plot
        plot = result.plot.bar(rot=0)
        plot.set(ylabel="Number of genes")
    elif (frequency_type == 'r') & (frequency_count == 'genes'):
        result = distribution_plot_re
        plot = result.plot.bar(rot=0)
        plot.set(ylabel="Relative number of genes")
    elif (frequency_type == 'log') & (frequency_count == 'genes'):
        result = distribution_plot
        plot = result.plot.bar(rot=0)
        plot.set_yscale('log')
        plot.set(ylabel="Number of genes")
    plot.set(xlabel="Contig size")

    return plot

def ftable_to_distribution(feature_table, organism='all', organism_name=False):
    """
    Take a feature table df as input and returns the distribuition of contig
    sizes
    """
    ft_distribution = feature_table.groupby(
        ['organism', 'genomic_accession']).agg({'locus_tag':'count'}).reset_index()
    ft_distribution = ft_distribution.groupby(
        ['organism', 'locus_tag']).agg({'genomic_accession':'count'}).reset_index()
    ft_distribution.columns = ['organism', 'contig_size', 'number_of_contigs']
    ft_distribution['number of genes'] = \
              ft_distribution['contig_size'] * ft_distribution['number_of_contigs']
    group = ft_distribution.groupby('organism')
    ft_distribution_2 = pd.DataFrame(columns=ft_distribution.columns)
    for _, ftables in group:
        ftables['genes_percent'] = ftables['number of genes']/ftables['number of genes'].sum()
        ftables['contig_percent'] =  \
                ftables['number_of_contigs']/ftables['number_of_contigs'].sum()
        ft_distribution_2 = ft_distribution_2.append(ftables)
        if organism == 'all':
            result = ft_distribution_2
        else:
            result = ft_distribution_2[ft_distribution_2.organism == organism]
            if organism_name:
                result = result[['organism', 'contig_size', 'number_of_contigs']]
            else:
                result = result[['contig_size', 'number_of_contigs']]
    return result
def uf_stats (uf):
    orthologous_group = uf.pivot_table(
        index='gid', columns='org', values='locus_tag', aggfunc='nunique').reset_index()
    orthologous_group.fillna(0, inplace=True)
    orthologous_group = orthologous_group.astype(int)
    orthologous_group['total'] = \
            (orthologous_group.Sspon + orthologous_group.SP803280 + orthologous_group.Sobic)
    orthologous_group['norg']  = orthologous_group.iloc[:, 1:5].apply(lambda x: np.sum(x > 0), axis=1)
    return orthologous_group

def box_plot_simulation(simulation='simulation', frequency_count='block', frequency_type='r', ylim=(0,600), xlim=(-1,10)):
    fig = plt.figure(dpi=100)
    if (frequency_count == 'block') & (frequency_type == 'r'):
        sns.boxplot(x='nlt',
                    y='blocks_percent',
                    data=simulation,
                    hue='org2',
                    linewidth=0.5,
                    fliersize=0.3)
        plt.ylabel('Relative block count')
    elif (frequency_count == 'block') & ((frequency_type == 'a') | (frequency_type == 'log')):
        sns.boxplot(x='nlt',
                    y='nblock',
                    data=simulation,
                    hue='org2',
                    linewidth=0.5,
                    fliersize=0.3)
        plt.ylabel('Block count')
    elif (frequency_count == 'genes') & (frequency_type == 'r'):
        sns.boxplot(x='nlt',
                    y='genes_percent',
                    data=simulation,
                    hue='org2',
                    linewidth=0.5,
                    fliersize=0.3)
        plt.ylabel('Relative genes count')
    elif (frequency_count == 'genes') & ((frequency_type == 'a') | (frequency_type == 'log')):
        sns.boxplot(x='nlt',
                    y='total_genes',
                    data=simulation,
                    hue='org2',
                    linewidth=0.5,
                    fliersize=0.3)
        plt.ylabel('Genes count')
    if frequency_type == 'log':
        plt.yscale('log')
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.legend()
    plt.xlabel('Genes per block')
    fig.show()
def contig_df_to_distribution(synteny_df, reference_organism='Sobic'):

    """
    Take a synteny df results as arument and create a distribution with
    percent using a model organism to compare.
    """
    block_count = synteny_df.groupby('block').agg(
        {'org1':'first', 'org2':'first','ga1':'first', 'lt1':'nunique', 'lt2':'nunique'})
    block_count.reset_index(inplace=True)
    block_count.columns = ['block', 'org1', 'org2','ga1', 'unique_lt1', 'unique_lt2']
    block_count['count_orthologous_lt'] = np.where(
        block_count.unique_lt1 <= block_count.unique_lt2,
        block_count.unique_lt1, block_count.unique_lt2)
    block_count = block_count.groupby(
        ['org1', 'org2', 'ga1', 'count_orthologous_lt']).agg({'block': 'count'}).reset_index()
    block_count.columns = ['org1', 'org2','ga1', 'nlt', 'nblock']
    group = block_count.groupby(['org1', 'org2', 'ga1'])
    bc_with_percent = pd.DataFrame(columns=['org1', 'org2','ga1', 'nlt', 'nblock'])
    for _, block_count_dfs  in group:
        block_count_dfs = block_count_dfs[(block_count_dfs.org1 == reference_organism)]
        block_count_dfs['blocks_percent'] = block_count_dfs.nblock/block_count_dfs.nblock.sum()
        block_count_dfs['total_genes'] = block_count_dfs.nblock * block_count_dfs.nlt
        block_count_dfs['genes_percent'] = block_count_dfs.total_genes/ block_count_dfs.total_genes.sum()
        bc_with_percent = bc_with_percent.append(block_count_dfs)
    bc_with_percent.total_genes = bc_with_percent.total_genes.astype(int)
    return bc_with_percent

def df_to_a(df,ft, Ref_org='Sobic', org_1='Sspon', org_2='SP803280'):

    def m(x):
        if isinstance(x, float):
            return 0
        return len(x.split(','))

    def block_max(num, block_size):
        import operator
        if isinstance(num, float):
            return np.nan
        dc = {}
        for idx in num.split(','):
            dc[idx] = block_size[block_size['block'] == int(idx)]['count'].values[0]
        return max(dc.items(), key=operator.itemgetter(1))[0]

    def block_max_len(num, block_size):
        if isinstance(num, float):
            return np.nan
        return block_size.loc[block_size.block.isin(set(num.split(','))), 'count'].max()

    def block_min(num, block_size):
        import operator
        if isinstance(num, float):
            return np.nan
        dc = {}
        for idx in num.split(','):
            dc[idx] = block_size[block_size['block'] == int(idx)]['count'].values[0]
        return min(dc.items(), key=operator.itemgetter(1))[0]

    def block_min_len(num, block_size):
        if isinstance(num, float):
            return np.nan
        return block_size.loc[block_size.block.isin(set(num.split(','))), 'count'].min()

    #block_size = df.groupby('block').size().reset_index()
    #block_size.columns = ['block', 'count']
    block_size = df.groupby('block').agg({'lt1':'nunique'}).reset_index()
    block_size.columns = ['block', 'count']


    dfbylt1 = df[df.org1 == Ref_org].groupby(
        ['org1', 'org2', 'lt1']).agg(
            {'ga2':'nunique', 'lt2':'nunique', 'block':lambda x: ",".join(
                x.unique().astype(str))}
        ).reset_index().sort_values(['org1', 'org2', 'lt1'])
    a = ft[ft.organism == Ref_org]
    a = a.merge(dfbylt1[dfbylt1.org2 == org_1 ], left_on=['locus_tag'], right_on=['lt1'], how='left')
    a.drop(columns=['lt1', 'org1', 'org2'], inplace=True)
    a = a.merge(dfbylt1[dfbylt1.org2 == org_2], left_on=['locus_tag'], right_on=['lt1'], how='left')
    a.drop(columns=['lt1', 'org1', 'org2'], inplace=True)
    a['Nblocs_Sspon'] = a.block_x.apply(lambda x: m(x))
    a['Nblocs_SP803280'] = a.block_y.apply(lambda x: m(x))
    a['block_dif_x'] = a.Nblocs_Sspon - a.Nblocs_SP803280
    a['ga_dif'] = a.ga2_x - a.ga2_y
    a['lt_dif'] = a.lt2_x - a.lt2_y
    a['block_y_max'] = a.block_y.map(lambda num: block_max(num, block_size))
    a['block_x_max'] = a.block_x.map(lambda num: block_max(num, block_size))
    a['len_block_y_max'] =  a.block_y.map(lambda x: block_max_len(x ,block_size))
    a['len_block_x_max'] =  a.block_x.map(lambda x: block_max_len(x ,block_size))
    a['block_y_min'] = a.block_y.map(lambda num: block_min(num, block_size))
    a['block_x_min'] = a.block_x.map(lambda num: block_min(num, block_size))
    a['len_block_y_min'] =  a.block_y.map(lambda x: block_min_len(x, block_size))
    a['len_block_x_min'] =  a.block_x.map(lambda x: block_min_len(x, block_size))
    a.columns = ['ftable_id', 'genomic_order', 'locus_tag', 'organism', 'assembly',
                 'genomic_accession', 'gstart', 'gend', 'strand', 'gid', f'oidx', f'ga2_{org_1}',
       f'lt2_{org_1}', f'block_{org_1}', f'ga2_{org_2}', f'lt2_{org_2}', f'block_{org_2}', 'Nblocs_Sspon',
       'Nblocs_SP803280', f'block_dif_{org_1}', 'ga_dif', 'lt_dif', f'block_{org_2}_max',
       f'block_{org_1}_max', f'len_block_{org_2}_max', f'len_block_{org_1}_max', f'block_{org_2}_min',
       f'block_{org_1}_min', f'len_block_{org_2}_min', f'len_block_{org_1}_min']

    return a
def a_to_distribution (a, block_filter=True, pivot=False):
    if block_filter:
        a = a[(a.Nblocs_Sspon != 0 ) & (a.Nblocs_SP803280 !=0)]
    h = a.groupby('len_block_SP803280_max').agg({'locus_tag':'nunique'}).reset_index()
    h.columns = ['blen', 'nblock']
    h['organism'] = 'SP803280'
    i = a.groupby('len_block_Sspon_max').agg({'locus_tag':'nunique'}).reset_index()
    i.columns = ['blen', 'nblock']
    i['organism'] = 'Sspon'
    h = h.append(i)
    if pivot:
        h = h.pivot(index='blen', columns='organism', values='nblock').reset_index().fillna(0)
    return h

def simulation_a_distribuition (a, ft, uf, simulation_rounds=10,contig_size=1000,
                                bfilter = True):

    def to_range(value=10):
        if isinstance(value, int):
            return list(range(value))
        elif isinstance(value,tuple):
            if len(value) == 1:
                return list(range(value[0]))
            elif len(value) == 2:
                return list(range(value[0], value[1]))
            elif len(value) == 3:
                return list(range(value[0], value[1], value[2]))
            else:
                return print('Insert a tuple with parameters to be a range input')
        else:
            return print('Insert a tuple as range argument or a int to make the simulation')

    times = list(to_range(simulation_rounds))
    simulation = pd.DataFrame(pd.DataFrame(columns=['organism','blen', 'SP803280', 'Sspon']))
    for x in times:
        a = random_distribution(ft,uf, contig_size)
        f = fake_ft(ft, a)
        df = make_df(f, uf)
        z = df_to_a(df,f)
        d = a_to_distribution(z, bfilter)
        d['simulation'] = x
        simulation = simulation.append(d)
    return simulation


def df_final(df, f):
    block_count = df.groupby('block').agg(
              {'org1':'first', 'org2':'first', 'lt1':'nunique', 'lt2':'nunique'})
    block_count.reset_index(inplace=True)
    block_count.columns = ['block', 'org1', 'org2', 'unique_lt1', 'unique_lt2']
    block_count['count_orthologous_lt'] = np.where(
              block_count.unique_lt1 <= block_count.unique_lt2,
              block_count.unique_lt1, block_count.unique_lt2)
    aaa= df.groupby('ga2').agg({'lt2': 'nunique'}).reset_index()
    tt = df.groupby('block').agg({'ga2': 'first'}).reset_index()
    bbb=tt.merge(aaa, left_on='ga2', right_on='ga2', how='left')
    ccc=block_count.merge(bbb, left_on='block', right_on='block', how='left')
    ccc=ccc[ccc.org1=='Sobic']
    blocks= ccc.groupby('ga2').agg({'org2': 'first','lt2':'first', 'count_orthologous_lt':['max', 'count']}).reset_index()
    blocks.columns = [ 'contig','organism','markers', 'bmax', 'bcount']
    c_size = f.groupby('genomic_accession').locus_tag.count().to_frame().reset_index()
    c_size.columns = ['contig', 'c_size']
    final = blocks.merge(c_size)
    return final

if __name__ == '__main__':
    pd.options.display.max_columns=10000
    (ft,uf) = load_tables('ftable.tsv','sacch.blast.e-5.uf.1.tsv')
    df =make_df(ft,uf)

