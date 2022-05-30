from rivagan import RivaGAN
import torch
#model = RivaGAN()
#model.fit("data/hollywood2", epochs=300, use_critic=True, use_adversary=True)
#model.save("./model_critic_adversary_64bit.pt")

#model.fit("data/hollywood2", epochs=300)
#model.save("./model_64bit.pt")

model = torch.load('./models/model.pt')
model.fit("data/moments-in-time", epochs=300, batch_size=12)
model.save("./moments_model.pt")
