from main_func import *


from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('which',type=int)
args = parser.parse_args()

# with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:   
#     res=[]
# for d in Path(r'/data/scsnake/ccta/').glob(r'S*/'):
d = sorted(Path(r'/data/scsnake/ccta/').glob(r'S?????/'))[args.which]
print('Processing: {}'.format(str(d)))
ct_dir = ''
res_dir = ''

for d1 in d.glob(r'originalDATA_name*/'):
    for d2 in d1.iterdir():
        ct_dir = str(d2)
        break
    break
for d1 in d.glob(r'centerlineDATA*/'):
    for d2 in d1.iterdir():
        res_dir = str(d2)
        break
    break

# %%pixie_debugger
ct = CtVolume()
ct.load_image_data(ct_dir + '/')

result = parse_results(res_dir + '/')


## benchmark
cor = Coronary(result['M1'])
s_mask, s_mpr = straighten_data_mask(
    ct, cor, output_dim=(100, 100), precision=(2, 2), output_spacing=0.2)

if 0:
    try:
        for vessel in result.keys():
    #             res.append(executor.submit(save_mask, ct, result, vessel, save_dir ='/data/scsnake/ccta/'+ct.id))        
            save_mask(ct, result, vessel, save_dir ='/data/scsnake/ccta/'+ d.name + '_' +ct.id)

    except Exception as ex:
        print(ex)
 
#         save_straightened_mask(ct, result, save_dir = '/data/scsnake/ccta/'+ct.id +'_straight' ,precision=(3,3))
#         save_mask(ct, result, save_dir ='/data/scsnake/ccta/_'+ct.id ,precision=(3,2,2))
#     for f in futures.as_completed(wait_for):
#         pass
