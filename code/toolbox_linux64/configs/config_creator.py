def config_creator(k):
    config = f"""<?xml version="1.0"?>
    <!-- Name attributes of 'att' nodes are not used, included just for reference.-->
    <config method='Mondrian' k='{k}'>

        <input filename='../../datasets/ring/ring.csv' separator=','/>
        <!-- If left blank, separator will be set as comma by default.-->
        <output filename='../anon_data/ring_mondrian/k{k}.csv' format ='genVals'/>
        <id> <!-- List of identifier attributes, if any, these will be excluded from the output -->
        </id>

        <qid>
            <att index='0' name ='A0'>
                <vgh value='[-6879.0: 6285.0]'>
                </vgh>
            </att>
            <att index='1' name ='A1'>
                <vgh value='[-7141.0: 6921.0]'>
                </vgh>
            </att>
            <att index='2' name ='A2'>
                <vgh value='[-7734.0: 7611.0]'>
                </vgh>
            </att>
            <att index='3' name ='A3'>
                <vgh value='[-6627.0: 7149.0]'>
                </vgh>
            </att>
            <att index='4' name ='A4'>
                <vgh value='[-7184.0: 6383.0]'>
                </vgh>
            </att>
            <att index='5' name ='A5'>
                <vgh value='[-6946.0: 6743.0]'>
                </vgh>
            </att>
            <att index='6' name ='A6'>
                <vgh value='[-7781.0: 6285.0]'>
                </vgh>
            </att>
            <att index='7' name ='A7'>
                <vgh value='[-6882.0: 6357.0]'>
                </vgh>
            </att>
            <att index='8' name ='A8'>
                <vgh value='[-7184.0: 7487.0]'>
                </vgh>
            </att>
            <att index='9' name ='A9'>
                <vgh value='[-7232.0: 6757.0]'>
                </vgh>
            </att>
            <att index='10' name ='A10'>
                <vgh value='[-7803.0: 7208.0]'>
                </vgh>
            </att>
            <att index='11' name ='A11'>
                <vgh value='[-7395.0: 6791.0]'>
                </vgh>
            </att>
            <att index='12' name ='A12'>
                <vgh value='[-7096.0: 6403.0]'>
                </vgh>
            </att>
            <att index='13' name ='A13'>
                <vgh value='[-7472.0: 7261.0]'>
                </vgh>
            </att>
            <att index='14' name ='A14'>
                <vgh value='[-7342.0: 7372.0]'>
                </vgh>
            </att>
            <att index='15' name ='A15'>
                <vgh value='[-7121.0: 6905.0]'>
                </vgh>
            </att>
            <att index='16' name ='A16'>
                <vgh value='[-7163.0: 7175.0]'>
                </vgh>
            </att>
            <att index='17' name ='A17'>
                <vgh value='[-8778.0: 6896.0]'>
                </vgh>
            </att>
            <att index='18' name ='A18'>
                <vgh value='[-7554.0: 5726.0]'>
                </vgh>
            </att>
            <att index='19' name ='A19'>
                <vgh value='[-6722.0: 7627.0]'>
                </vgh>
            </att>
        </qid>


        <sens>
            <att index='20' name='class'/>
        </sens>
    </config>
    """
    f = open(f"mondrian{k}.xml", "w+")
    f.write(config)
    f.close()

for k in range(100, 7400,250):
    config_creator(k)
