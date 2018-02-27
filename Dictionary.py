
# coding: utf-8

# - HGNC JSON File: hgnc_complete_set.json
# - ChEBI SDF File: ChEBI_complete_3star.sdf
# - ChEBI OWL File: chebi.owl
# - GOBP  OWL File: go.owl
# - MESHD XML File: meshd.xml
# - MGI to HGNC   : HOM_MouseHumanSequence.rpt
# - EGID to HGNC  : [not used] gene_info (Very large file)

# In[29]:


import json, pprint, csv, re
from collections import Counter
import xml.etree.ElementTree as ET


# In[30]:


def loadSentences(filename):
    f = open(filename, encoding="utf8")
    reader = csv.DictReader(f, delimiter='\t')
    sentences = [row for row in reader]
    return sentences


# In[31]:


def createIndexFromDict(HGNCDict):
    HGNCIndex = {}
    for key, val in HGNCDict.items():
        for name in val['all_name']:
            if name in HGNCIndex:
                HGNCIndex[name].append(key)
            else:
                HGNCIndex[name] = [key]
    return HGNCIndex


# In[32]:


def getHGNCDictIndex():
    f = open("./dictionary/hgnc_complete_set.json")
    HGNCStr = f.read()
    HGNC = json.loads(HGNCStr)
    # pprint.pprint(HGNC)
    docs = HGNC['response']['docs']
    # pprint.pprint(docs[25])
    HGNCDict = dict()
    HGNCID2Symbol = dict()
    EGID2HGNC = dict()
    for file in docs:
        hid = file['hgnc_id'][5:]
        symb = file['symbol']
        HGNCID2Symbol[hid] = symb
        if 'entrez_id' in file:
            egid = file['entrez_id']
            EGID2HGNC[egid] = symb
        HGNCDict[symb] = {
            'symbol': file['symbol'],
            'name': file['name'],
            'alias_name': file.get('alias_name',[]),
            'alias_symbol': file.get('alias_symbol',[]),
            'cosmic': file.get('cosmic',''),
            'prev_name': file.get('prev_name',[]),
            'prev_symbol': file.get('prev_symbol',[]),
        }
        HGNCDict[symb]['all_name'] = [symb, file.get('name',''), file.get('cosmic','')]
        HGNCDict[symb]['all_name'].extend(file.get('alias_name',[]))
        HGNCDict[symb]['all_name'].extend(file.get('alias_symbol',[]))
        HGNCDict[symb]['all_name'].extend(file.get('prev_name',[]))
        HGNCDict[symb]['all_name'].extend(file.get('prev_symbol',[]))
        HGNCDict[symb]['all_name'] = [x.lower() for x in HGNCDict[symb]['all_name']]
        HGNCDict[symb]['all_name'] = list(set([x for x in HGNCDict[symb]['all_name'] if x != '']))
#         HGNCDict[symb]['all_name'] = list(set([x.replace('-', ' ') for x in HGNCDict[symb]['all_name'] if x != '']))
    del HGNC

    HGNCIndex = createIndexFromDict(HGNCDict)
    countIndex = [len(val) for key, val in HGNCIndex.items()]
    countFreq = Counter(countIndex)
    f.close()
    
    print('--------Successfully processed HGNCDictIndex--------')
    print('No. Entities:', len(HGNCDict))
    print('Index Information:', countFreq)
    
    return HGNCDict, HGNCIndex, HGNCID2Symbol, EGID2HGNC


# In[33]:


def getChEBIDictIndexOld():
    f = open('ChEBI_complete_3star.sdf')
    ChEBIDict = dict()
    lines = f.readlines()
    newObject = None
    i = 0
    while i < len(lines):
        if lines[i].startswith('> <ChEBI Name>'):
            if newObject is not None: # Collect the previous object before starting the new object
                newObject['all_name'] = [newObject['symbol']]
                newObject['all_name'].extend(newObject.get('synonyms', []))
                if 'iupac' in newObject:
                    newObject['all_name'].append(newObject['iupac'])
                newObject['all_name'] = [x.lower() for x in newObject['all_name']]
                newObject['all_name'] = list(set(newObject['all_name']))
                ChEBIDict[newObject['symbol']] = newObject
            i += 1
            newObject = {'symbol': lines[i].strip()}
        elif lines[i].startswith('> <IUPAC Names>'):
            i += 1
            newObject['iupac'] = lines[i].strip()
        elif lines[i].startswith('> <Synonyms>'):
            synset = []
            i += 1
            while True:
                if lines[i].strip() == '':
                    break
                synset.append(lines[i].strip())
                i += 1
            newObject['synonyms'] = synset
        i += 1
   
    ChEBIIndex = createIndexFromDict(ChEBIDict)
    countIndex = [len(val) for key, val in ChEBIIndex.items()]
    countFreq = Counter(countIndex)
    f.close()
    
    print('--------Successfully processed ChEBIDictIndex--------')
    print('No. Entities:', len(ChEBIDict))
    print('Index Information:', countFreq)
    
    return ChEBIDict, ChEBIIndex


# In[34]:


def getChEBIDictIndex():
    tree = ET.parse('./dictionary/chebi.owl')
    root = tree.getroot()
    ChEBIDict = dict()
    for cl in root.findall('{http://www.w3.org/2002/07/owl#}Class'):
        if cl.find('{http://www.w3.org/2000/01/rdf-schema#}label') is not None:
#             if cl.find('{http://purl.obolibrary.org/obo/chebi/}formula') is None:
#                 continue
            if cl.find('{http://www.geneontology.org/formats/oboInOwl#}inSubset') is None or cl.find('{http://www.geneontology.org/formats/oboInOwl#}inSubset').attrib['{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource'] != 'http://purl.obolibrary.org/obo/chebi#3_STAR':
                continue # Select only 3 STAR chemicals
            newObject = {'uri':cl.attrib['{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about']}
            newObject['symbol'] = cl.find('{http://www.w3.org/2000/01/rdf-schema#}label').text.strip()
            newObject['has_broad_synonym'] = [n.text.strip() for n in cl.findall('{http://www.geneontology.org/formats/oboInOwl#}hasBroadSynonym')]
            newObject['has_exact_synonym'] = [n.text.strip() for n in cl.findall('{http://www.geneontology.org/formats/oboInOwl#}hasExactSynonym')]
            newObject['has_narrow_synonym'] = [n.text.strip() for n in cl.findall('{http://www.geneontology.org/formats/oboInOwl#}hasNarrowSynonym')]
            newObject['has_related_synonym'] = [n.text.strip() for n in cl.findall('{http://www.geneontology.org/formats/oboInOwl#}hasRelatedSynonym')]
            newObject['definition'] = cl.find('{http://purl.obolibrary.org/obo/}IAO_0000115').text.strip() if cl.find('{http://purl.obolibrary.org/obo/}IAO_0000115') is not None else ''
            newObject['all_name'] = [newObject['symbol']]
            newObject['all_name'].extend(newObject['has_broad_synonym'])
            newObject['all_name'].extend(newObject['has_exact_synonym'])
            newObject['all_name'].extend(newObject['has_narrow_synonym'])
            newObject['all_name'].extend(newObject['has_related_synonym'])
            newObject['all_name'] = [x.lower() for x in newObject['all_name']]
#             newObject['all_name'] = [x.lower().replace('-',' ') for x in newObject['all_name']]
            newObject['all_name'] = list(set(newObject['all_name']))
            ChEBIDict[newObject['symbol']] = newObject

    ChEBIIndex = createIndexFromDict(ChEBIDict)
    countIndex = [len(val) for key, val in ChEBIIndex.items()]
    countFreq = Counter(countIndex)
    
    print('--------Successfully processed ChEBIDictIndex--------')
    print('No. Entities:', len(ChEBIDict))
    print('Index Information:', countFreq)
    
    return ChEBIDict, ChEBIIndex


# In[35]:


def getGOBPDictIndex():
    tree = ET.parse('./dictionary/go.owl')
    root = tree.getroot()
    GOBPDict = dict()
    for cl in root.findall('{http://www.w3.org/2002/07/owl#}Class'):
        if cl.find('{http://www.w3.org/2000/01/rdf-schema#}label') is not None:
            if cl.find('{http://www.geneontology.org/formats/oboInOwl#}hasOBONamespace') is None or cl.find('{http://www.geneontology.org/formats/oboInOwl#}hasOBONamespace').text != 'biological_process':
                continue # Select only biological process
            newObject = {'uri':cl.attrib['{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about']}
            newObject['symbol'] = cl.find('{http://www.w3.org/2000/01/rdf-schema#}label').text.strip()
            newObject['has_broad_synonym'] = [n.text.strip() for n in cl.findall('{http://www.geneontology.org/formats/oboInOwl#}hasBroadSynonym')]
            newObject['has_exact_synonym'] = [n.text.strip() for n in cl.findall('{http://www.geneontology.org/formats/oboInOwl#}hasExactSynonym')]
            newObject['has_narrow_synonym'] = [n.text.strip() for n in cl.findall('{http://www.geneontology.org/formats/oboInOwl#}hasNarrowSynonym')]
            newObject['has_related_synonym'] = [n.text.strip() for n in cl.findall('{http://www.geneontology.org/formats/oboInOwl#}hasRelatedSynonym')]
            newObject['definition'] = cl.find('{http://purl.obolibrary.org/obo/}IAO_0000115').text.strip() if cl.find('{http://purl.obolibrary.org/obo/}IAO_0000115') is not None else ''
            newObject['all_name'] = [newObject['symbol']]
            newObject['all_name'].extend(newObject['has_broad_synonym'])
            newObject['all_name'].extend(newObject['has_exact_synonym'])
            newObject['all_name'].extend(newObject['has_narrow_synonym'])
            newObject['all_name'].extend(newObject['has_related_synonym'])
            newObject['all_name'] = [x.lower() for x in newObject['all_name']]
#             newObject['all_name'] = [x.lower().replace('-',' ') for x in newObject['all_name']]
            newObject['all_name'] = list(set(newObject['all_name']))
            GOBPDict[newObject['symbol']] = newObject

    GOBPIndex = createIndexFromDict(GOBPDict)
    countIndex = [len(val) for key, val in GOBPIndex.items()]
    countFreq = Counter(countIndex)
    
    print('--------Successfully processed GOBPDictIndex--------')
    print('No. Entities:', len(GOBPDict))
    print('Index Information:', countFreq)
    
    return GOBPDict, GOBPIndex


# In[36]:


def getMESHDictIndex():
    tree = ET.parse('./dictionary/meshd.xml')
    root = tree.getroot()
    MESHDict = dict()
    for desc in root.findall('DescriptorRecord'):
        if desc.find('TreeNumberList') == None or desc.find('TreeNumberList').find('TreeNumber').text[0] != 'C':
            continue
        symbol = desc.find('DescriptorName').find('String').text
        preferredConcept = []
        relatedConcept = []
        conceptList = desc.find('ConceptList').findall('Concept')
        for conc in conceptList:
            if conc.attrib['PreferredConceptYN'] == 'Y':
                preferredConcept.extend([t.find('String').text for t in conc.find('TermList').findall('Term')])
            else:
                relatedConcept.extend([t.find('String').text for t in conc.find('TermList').findall('Term')])
        allName = [symbol]
        allName.extend(preferredConcept)
        allName.extend(relatedConcept)
        MESHDict[symbol] = {
            'symbol': symbol,
            'preferred_concept': list(set(preferredConcept)),
            'related_concept': list(set(relatedConcept)),
            'all_name': list(set([x.lower() for x in allName]))
#             'all_name': list(set([x.lower().replace('-', ' ') for x in allName]))
        }
   
    MESHIndex = createIndexFromDict(MESHDict)
    countIndex = [len(val) for key, val in MESHIndex.items()]
    countFreq = Counter(countIndex)
    
    print('--------Successfully processed MESHDictIndex--------')
    print('No. Entities:', len(MESHDict))
    print('Index Information:', countFreq)
    
    return MESHDict, MESHIndex


# In[37]:


def getPossibleMatches(text):
    text = text.lower()
    resultsList = []
    if text in HGNCIndex:
        resultsList.extend(['HGNC:'+x if all([a not in x for a in [' ', '(', ')', '+', '-']]) else 'HGNC:"'+x+'"' for x in HGNCIndex[text]])
    if text in ChEBIIndex:
        resultsList.extend(['CHEBI:'+x if all([a not in x for a in [' ', '(', ')', '+', '-']]) else 'CHEBI:"'+x+'"' for x in ChEBIIndex[text]])
    if text in GOBPIndex:
        resultsList.extend(['GOBP:'+x if all([a not in x for a in [' ', '(', ')', '+', '-']]) else 'GOBP:"'+x+'"' for x in GOBPIndex[text]])
    if text in MESHIndex:
        resultsList.extend(['MESHD:'+x if all([a not in x for a in [' ', '(', ')', '+', '-']]) else 'MESHD:"'+x+'"' for x in MESHIndex[text]])
    return resultsList


# In[38]:


def getMGI2HGNCDict():
    sentences = loadSentences('./dictionary/HOM_MouseHumanSequence.rpt')
    groupSentence = dict()
    MGI2HGNC = dict()
    for row in sentences:
        if row['HomoloGene ID'] not in groupSentence:
            aMap = {'MGI':[],'HGNC':[]}
        else:
            aMap = groupSentence[row['HomoloGene ID']]
        if row['Common Organism Name'].strip() == 'mouse, laboratory':
            aMap['MGI'].append(row['Symbol'].strip())
        elif row['Common Organism Name'].strip() == 'human':
            aMap['HGNC'].append(row['Symbol'].strip())
        groupSentence[row['HomoloGene ID']] = aMap
    for hid in groupSentence.keys():
        aMap = groupSentence[hid]
        if len(aMap['MGI']) > 0 and len(aMap['HGNC']) > 0 and len(aMap['MGI']) + len(aMap['HGNC']) > 2:
            pass
#             print(hid)
        if len(aMap['MGI']) == 1 and len(aMap['HGNC']) == 1:
            MGI2HGNC[aMap['MGI'][0]] = aMap['HGNC'][0]
    return MGI2HGNC


# In[39]:


def getEGID2HGNCDict():
    EGID2HGNC = dict()
    sentences = loadSentences('gene_info')
    for s in sentences:
        if 'HGNC' in s['dbXrefs']:
            egid = s['GeneID']
            searchObj = re.search( r'HGNC:HGNC(\d+)', s['dbXrefs'])
            EGID2HGNC[egid] = HGNCID2Symbol[searchObj.group()]
    return EGID2HGNC


# In[ ]:


def testDict():
    print('\n----HGNCDict----')
    print(HGNCDict['F2'])  
    print(HGNCIndex['tc10'])
    print(HGNCIndex['tnf'])
    print('\n----ChEBIDict----')
    print(ChEBIDict['nitric oxide'])    
    print(ChEBIIndex['insulin'])
    print('\n----GOBPDict----')
    print(GOBPDict["osteoclast differentiation"])
    print(GOBPIndex["wound healing"])
    print('\n----MESHDict----')
    print(MESHDict['Hyperoxaluria, Primary'])
    print(MESHIndex['oxalosis'])
    print('\n')


# In[ ]:


HGNCDict, HGNCIndex, HGNCID2Symbol, EGID2HGNC = getHGNCDictIndex()
ChEBIDict, ChEBIIndex = getChEBIDictIndex()
GOBPDict, GOBPIndex = getGOBPDictIndex()
MESHDict, MESHIndex = getMESHDictIndex()
MGI2HGNC = getMGI2HGNCDict()


# In[ ]:


# testDict()
# print(GOBPDict['cell migration'])
# print(getPossibleMatches('apoptotic process'))
# print(MGI2HGNC['Pde3b'])
# print(EGID2HGNC['9687'])

