from torch.utils.data import DataLoader
from utils import *
from network.Network import *

from utils.load_test_setting import *
from utils.linear_block_code import *


'''
test
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = Network(H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion)
EC_path = result_folder + "models/EC_" + str(model_epoch) + ".pth"
network.load_model_ed(EC_path)

test_dataset = MBRSDataset(os.path.join(dataset_path, "test"), H, W)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

print("\nStart Testing : \n\n")

test_result = {
    "error_rate": 0.0,
    "psnr": 0.0,
    "ssim": 0.0
}

start_time = time.time()

saved_iterations = np.random.choice(np.arange(len(test_dataset)), size=save_images_number, replace=False)
saved_all = None

# generate message encoded by lbc
msgstr = 'bilibili@copyright'
m = stringToBitArray(msgstr, length=8)
m = np.array(m).reshape(-1, 4).astype(int)
G = np.array([[1, 1, 1, 1, 0, 0, 0],
              [1, 1, 0, 0, 1, 0, 0],
              [1, 0, 1, 0, 0, 1, 0],
              [0, 1, 1, 0, 0, 0, 1]])
lbc = LinearBlockCode()
lbc.setG(G)
c = lbc.c(m)
c = torch.Tensor(c.reshape(-1))
message = torch.Tensor(np.random.choice([0, 1], (batch_size, message_length))).to(device)
if len(c) > message_length:
    print("the lbc message should not exceed message length")
    raise ValueError
message[:, :len(c)] = c

num = 0
for i, images in enumerate(test_dataloader):
    image = images.to(device)

    '''
    test
    '''
    network.encoder_decoder.eval()
    network.discriminator.eval()

    with torch.no_grad():
        # use device to compute
        images, messages = images.to(network.device), message.to(network.device)

        encoded_images = network.encoder_decoder.module.encoder(images, messages)
        encoded_images = images + (encoded_images - image) * strength_factor
        noised_images = network.encoder_decoder.module.noise([encoded_images, images])

        decoded_messages = network.encoder_decoder.module.decoder(noised_images)

        # psnr
        psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2).item()

        # ssim
        ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean").item()

    '''
    decoded message error rate
    '''
    decoded_messages = decoded_messages.gt(0.5)
    
    # lbc error correction
    for b in range(batch_size):
        rs = decoded_messages[b, :len(c)]
        rs = rs.reshape(-1, lbc.n()).cpu().numpy()
        cs = np.zeros_like(rs)
        for j in range(len(rs)):
            cs[j] = lbc.syndromeDecode(rs[j])
        decoded_messages[b, :len(c)] = torch.Tensor(cs.reshape(-1)).to(device)
        
    ground_messages = messages.gt(0.5)
    error_bits = decoded_messages != ground_messages
    error_rate = error_bits.sum() / error_bits.numel()

    # error_rate = network.decoded_message_error_rate_batch(messages, decoded_messages)

    result = {
        "error_rate": error_rate,
        "psnr": psnr,
        "ssim": ssim,
    }

    for key in result:
        test_result[key] += float(result[key])

    num += 1

    if i in saved_iterations:
        if saved_all is None:
            saved_all = get_random_images(image, encoded_images, noised_images)
        else:
            saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)

    '''
    test results
    '''
    content = "Image " + str(i) + " : \n"
    for key in test_result:
        content += key + "=" + str(result[key]) + ","
    content += "\n"

    with open(test_log, "a") as file:
        file.write(content)

    print(content)

'''
test results
'''
content = "Average : \n"
for key in test_result:
    content += key + "=" + str(test_result[key] / num) + ","
content += "\n"

with open(test_log, "a") as file:
    file.write(content)

print(content)
# save_images(saved_all, "test", result_folder + "images/", resize_to=(W, H))
