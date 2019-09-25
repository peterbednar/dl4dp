from conllutils import shuffled_stream

def train(model, optimizer, params):

    for epoch in range(params.max_epoch):
        print(f"epoch: {epoch + 1}")

        for batch in shuffled_stream(params.train_data, total_size=len(params.train_data), batch_size=params.batch_size):
            optimizer.zero_grad()
            arc_loss, label_loss = model.loss(batch)

            loss = arc_loss + label_loss
            loss.backward()
            optimizer.step()

            print(f"{arc_loss.item()}")
