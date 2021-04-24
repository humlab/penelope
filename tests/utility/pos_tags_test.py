from penelope.utility import pos_tags


def test_pos_tags():

    all_tags = set('AB|DT|HA|HD|HP|HS|IE|IN|JJ|KN|MAD|MID|NN|PAD|PC|PL|PM|PN|PP|PS|RG|RO|SN|UO|VB'.split('|'))

    suc_tags = pos_tags.PD_SUC_PoS_tags

    assert all_tags.difference(set(suc_tags.tag.tolist())) == set()
    assert all_tags.difference(set(pos_tags.PoS_Tag_Schemes.SUC.tags)) == set()

    suc_schema: pos_tags.PoS_Tag_Scheme = pos_tags.PoS_Tag_Schemes.SUC

    tags = suc_schema.exclude("Delimiter")
    assert set(tags).difference(all_tags - {'MID', 'MAD', 'PAD'}) == set()

    tags = suc_schema.exclude(['MID', 'MAD', 'PAD'])
    assert set(tags).difference(all_tags - {'MID', 'MAD', 'PAD'}) == set()

    tags = suc_schema.exclude(['MID', 'MAD', 'PAD', "Delimiter"])
    assert set(tags).difference(all_tags - {'MID', 'MAD', 'PAD'}) == set()

    tags = suc_schema.exclude(['NN', "Delimiter", ['VB']])
    assert set(tags).difference(all_tags - {'MID', 'MAD', 'PAD', 'NN', 'VB'}) == set()

    tags = suc_schema.exclude(['NN|JJ', "Delimiter", ['VB']])
    assert set(tags).difference(all_tags - {'MID', 'MAD', 'PAD', 'NN', 'VB', 'JJ'}) == set()
