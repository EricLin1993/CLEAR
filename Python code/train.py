import torch
import torch.optim as optim
import torch.nn.functional as F

from utils import Dataset
from model import complex_mse_loss,CLEAR
import time
from torch.utils.data import DataLoader,random_split

def train_model(model, train_loader,val_loader, criterion, optimizer, num_epochs=25,scheduler=None,patience=5,model_result_path=None):
    model.train()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        if early_stop:
            print('Early Stop Training !!!')
            break
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}, Learning Rate: {current_lr}")

        star_time=time.perf_counter()
        train_loss = 0.0
        for inputs, targets,sampled_signal,mask in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            sampled_signal = sampled_signal.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            outputs = model(inputs,sampled_signal,mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() 
        
        train_loss = train_loss / len(train_loader)

        model.eval()  
        val_loss = 0.0
        with torch.no_grad():  
            for inputs, targets,sampled_signal,mask in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                sampled_signal = sampled_signal.to(device)
                mask = mask.to(device)   

                output = model(inputs,sampled_signal,mask)
                loss = criterion(output, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        end_time  = time.perf_counter()
        training_time = end_time-star_time
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, Training_time :{(training_time):.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_result_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve > patience:
                print(f"Validation loss did not improve for {patience} epochs. Early stopping triggered.")
                early_stop = True 
        
        if early_stop:
            model.load_state_dict(torch.load(model_result_path))
            print("Best model weights loaded.")

def lr_lambda(epoch):
        if epoch < 20:
            return 10  
        else:
            return 1   

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    directory = '/your simulation data path'

    input_key   = 'f_2d_nus_2c'  
    target_key  = 'f_2d_2c'   
    sampled_key = 'fid_2d_nus_2c'
    mask_key    = 'mask_2c'
    model_result_path = './your_model_result_path.pth' 

    dataset = Dataset(directory, input_key, target_key, sampled_key,mask_key)
    print('Number of Simulated Dataset:',len(dataset))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = 2  
    model_dim = 32
    num_heads = 2
    num_layers = 2  
    conv_kernel_size = (17,17)  
    dropout = 0.1
    learning_rate = 1e-4
    num_epochs = 30

    model = CLEAR(input_dim = input_dim, model_dim = model_dim, num_heads = num_heads, num_layers = num_layers, ff_expansion_factor=4,conv_kernel_size = conv_kernel_size, dropout=dropout)
    

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = complex_mse_loss
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_model(model, train_loader,val_loader, criterion, optimizer, num_epochs=num_epochs,scheduler=scheduler,patience=10,model_result_path=model_result_path)
