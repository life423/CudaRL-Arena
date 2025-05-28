extends Node

# This runs headless and provides training data via TCP
var server := TCPServer.new()
var clients := []
var port := 9999
var training_data := {}

func _ready():
    if server.listen(port) != OK:
        print("Failed to start training server on port ", port)
        return
        
    print("Training server listening on port ", port)
    
    # Set up training data structure
    training_data = {
        "episode": 0,
        "step": 0,
        "reward": 0.0,
        "total_reward": 0.0,
        "state": [],
        "action": -1,
        "done": false
    }

func _process(_delta):
    # Accept new connections
    if server.is_connection_available():
        var client = server.take_connection()
        clients.append(client)
        print("Training client connected. Total clients: ", clients.size())
    
    # Handle existing clients
    for i in range(clients.size() - 1, -1, -1):
        var client = clients[i]
        if client.get_status() != StreamPeerTCP.STATUS_CONNECTED:
            clients.remove_at(i)
            print("Training client disconnected. Total clients: ", clients.size())
            continue
            
        if client.get_available_bytes() > 0:
            _handle_client_data(client)

func _handle_client_data(client: StreamPeerTCP):
    var data = client.get_string(client.get_available_bytes())
    var lines = data.split("\n")
    
    for line in lines:
        if line.strip_edges() == "":
            continue
            
        var json = JSON.new()
        var parse_result = json.parse(line)
        if parse_result != OK:
            continue
            
        var message = json.data
        _process_training_message(message)

func _process_training_message(message: Dictionary):
    if message.has("type"):
        match message["type"]:
            "state_update":
                _update_training_state(message)
            "episode_start":
                _start_new_episode(message)
            "episode_end":
                _end_episode(message)

func _update_training_state(message: Dictionary):
    if message.has("episode"): training_data["episode"] = message["episode"]
    if message.has("step"): training_data["step"] = message["step"]
    if message.has("reward"): training_data["reward"] = message["reward"]
    if message.has("total_reward"): training_data["total_reward"] = message["total_reward"]
    if message.has("state"): training_data["state"] = message["state"]
    if message.has("action"): training_data["action"] = message["action"]
    if message.has("done"): training_data["done"] = message["done"]
    
    # Print progress occasionally
    if training_data["step"] % 100 == 0:
        print("Episode %d, Step %d, Reward: %.2f, Total: %.2f" % [
            training_data["episode"],
            training_data["step"], 
            training_data["reward"],
            training_data["total_reward"]
        ])

func _start_new_episode(message: Dictionary):
    training_data["episode"] = message.get("episode", training_data["episode"] + 1)
    training_data["step"] = 0
    training_data["total_reward"] = 0.0
    print("Starting episode ", training_data["episode"])

func _end_episode(message: Dictionary):
    var final_reward = message.get("total_reward", training_data["total_reward"])
    print("Episode %d completed - Total reward: %.2f" % [training_data["episode"], final_reward])

func _exit_tree():
    server.stop()
    for client in clients:
        client.disconnect_from_host()
