def config_creator(k):
    config = f"""<?xml version="1.0"?>
    <!-- Name attributes of 'att' nodes are not used, included just for reference.-->
    <config method='Mondrian' k='{k}'>

        <input filename='../../datasets/birth_control/cmc_upsampled.csv' separator=','/>
        <!-- If left blank, separator will be set as comma by default.-->
        <output filename='../anon_data/birth_mondrian/k{k}.csv' format ='genVals'/>
        <id> <!-- List of identifier attributes, if any, these will be excluded from the output -->
        </id>

        <qid>
            <att index='0' name ='age'>
                <vgh value='[16:49]'>
                </vgh>
            </att>
            <att index='1' name ='wife_ed'>
                <vgh value='[1:4]'>
                </vgh>
            </att>
            <att index='2' name ='husb_ed'>
                <vgh value='[1:4]'>
                </vgh>
            </att>
            <att index='3' name ='no_kids'>
                <vgh value='[0:16]'>
                </vgh>
            </att>
            <att index='4' name ='wife_rel'>
                <vgh value='[0:1]'>
                </vgh>
            </att>
            <att index='5' name ='wife_works'>
                <vgh value='[0:1]'>
                </vgh>
            </att>
            <att index='6' name ='husb_occupation'>
                <vgh value='[1:4]'>
                </vgh>
            </att>
            <att index='7' name ='SOL_index'>
                <vgh value='[1:4]'>
                </vgh>
            </att>
            <att index='8' name ='media_exp'>
                <vgh value='[0:1]'>
                </vgh>
            </att>
        </qid>

        <sens>
            <att index='9' name='class'/>
        </sens>
    </config>
    """
    f = open(f"mondrian{k}.xml", "w+")
    f.write(config)
    f.close()

for k in list(range(2,20,2)) + list(range(2, 736,15)) + [1887]:
    config_creator(k)
