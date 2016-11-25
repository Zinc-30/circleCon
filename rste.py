import numpy as np
import math

def process_data(users, items, ratings):
    num_of_items = get_number(items, "::")
    num_of_users = get_number(users, ":")
    ratingMatrix = create_rating_matrix(num_of_users, num_of_items,ratings)
    socialMatrix = create_social_matrix(num_of_users, users)
    model_training(int(num_of_users), ratingMatrix, int(num_of_items), socialMatrix, 5, 100, 0.5, 0.1)
    # model_training(num_of_users, ratingMatrix, num_of_items, socialMatrix, latent_feature_dimention, num_of_iterations,alpha, theta)
    # model_training(2, ratingMatrix, 2, socialMatrix, 4, 1000, 0.5, 0.1)


def create_rating_matrix(num_of_users, num_of_items, ratings):
    ratingMatrix = np.zeros((int(num_of_users), int(num_of_items)))
    file = open(ratings,'r')
    for line in file:
        user = line.split("::")[0]
        item = line.split("::")[1]
        score = line.split("::")[2]
        ratingMatrix[int(user) - 1][int(item) - 1] = float(score)/5.0
    # return [[0.5, 0.8], [0.9, 0.1]]
    return ratingMatrix


def create_social_matrix(num_of_users, users):
    socialMatrix = np.zeros((int(num_of_users), int(num_of_users)))
    file = open(users, 'r')
    print "enter"
    for line in file:
        user = line.split(":")[0]
        friends = line.split(":")[1][1:len(line.split(":")[1])-3]
        friend_list = friends.split(",")
        if friend_list:
            for f in range(0,len(friend_list)-1):
                socialMatrix[int(user) - 1][int(friend_list[f])-1] = 1
    # return [[0, 1], [1, 0]]
    return socialMatrix


def get_number(data_path, delimitation):
    file = open(data_path, 'r')
    lastline = 0
    for line in file:
        lastline = line
    return lastline.split(delimitation)[0]


def model_training(num_of_users, ratingMatrix, num_of_items, socialMatrix, latent_feature_dimention, num_of_iterations, alpha, theta):
    log = open('log.txt', 'w')
    log.write('output information:\n')

    print "#Users: " + str(num_of_users)
    print "#Items: " + str(num_of_items)
    log.write("#Users: " + str(num_of_users) + '\n')
    log.write("#Items: " + str(num_of_items)+ '\n')
    log.write("#latent feature dimention: " + str(latent_feature_dimention) + '\n')
    log.write("#alpha: " + str(alpha) + '\n')
    log.write("#theta: " + str(theta) + '\n')

    # l: dimention of user/item latent feature vector
    U = np.random.randn(latent_feature_dimention, num_of_users) # l x m
    V = np.random.randn(num_of_items, latent_feature_dimention) # n x l
    old_U = U
    old_V = V

    #number of iterations to update U,V
    for iter in range(0, num_of_iterations):
        #calculate cost funciton
        # if iter == 0:
        #     difference = 0
        #     for u in range(0, num_of_users-1):
        #         user_latent = U[:,u]
        #         for i in range(0, num_of_items-1):
        #             if ratingMatrix[u][i] != 0:
        #                 item_latent = V[i,:]
        #                 latent_pred = user_latent.dot(item_latent) * alpha
        #                 friends_rating = 0
        #                 for f in (0, num_of_users-1):
        #                     if socialMatrix[u][f] == 1:
        #                         friends_rating += U[:,f].dot(item_latent)
        #                 friends_pred = (1-alpha)*friends_rating
        #                 rating_pred = sigmoid(friends_pred + latent_pred)
        #                 difference += 0.5 * math.pow(ratingMatrix[u][i] - rating_pred, 2)
        #     cost_funciton = difference + 0.5*frobenius_norm(U, latent_feature_dimention, num_of_users) + 0.5*frobenius_norm(V, num_of_items, latent_feature_dimention)
        #     print "first cost: " + str(cost_funciton)
        #     log.write("first cost: " + str(cost_funciton) + '\n')
        #
        # print "enter iteration: " + str(iter)
        # log.write("enter iteration: " + str(iter) + '\n')

        # update Ui
        for u in range(0, num_of_users-1):
            print "iteration: " + str(iter) + " update user " + str(u)
            log.write("iteration: " + str(iter) + " update user " + str(u) + '\n')
            user_latent = U[:, u]
            first_term = 0
            for i in range(0, num_of_items - 1):
                if ratingMatrix[u][i] != 0:
                    item_latent = V[i, :]
                    latent_pred = user_latent.dot(item_latent) * alpha
                    friends_rating = 0
                    for f in (0, num_of_users-1):
                        if socialMatrix[u][f] == 1:
                            friends_rating += U[:, f].dot(item_latent)
                    friends_pred = (1 - alpha) * friends_rating
                    pred = friends_pred + latent_pred
                    first_term += derivative_sigmoid(pred) * item_latent * (sigmoid(pred) - ratingMatrix[u][i])

            second_term = 0
            for trust_u in range(0, num_of_users-1): # users who trust u
                if socialMatrix[trust_u][u] == 1:
                    trust_user_latent = U[:,trust_u]
                    for i in range(0, num_of_items - 1):
                        if ratingMatrix[trust_u][i] != 0:
                            item_latent = V[i, :]
                            latent_pred = trust_user_latent.dot(item_latent) * alpha
                            friends_rating = 0
                            for f in (0,num_of_users-1):
                                if socialMatrix[trust_u][f] == 1:
                                    friends_rating += U[:,f].dot(item_latent)
                            friends_pred = (1 - alpha) * friends_rating
                            pred = friends_pred + latent_pred
                            second_term += derivative_sigmoid(pred)*(sigmoid(pred)-ratingMatrix[trust_u][i])*socialMatrix[trust_u][u]*item_latent
            derivative_Ui = alpha*first_term + (1-alpha)*second_term + user_latent
            old_U[:,u] = user_latent - theta*derivative_Ui

        #update Vj
        for i in range(0, num_of_items - 1):
            print "iteration: " + str(iter) + " update item " + str(i)
            log.write("iteration: " + str(iter) + " update item " + str(i)+ '\n')
            item_latent = V[i, :]
            sum = 0
            sum_of_final_term = 0
            for u in range(0, num_of_users - 1):
                if ratingMatrix[u][i] != 0:
                    user_latent = U[:, u]
                    latent_pred = user_latent.dot(item_latent) * alpha
                    friends_rating = 0
                    for f in (0, num_of_users-1):
                        if socialMatrix[u][f] == 1:
                            friends_rating += U[:, f].dot(item_latent)
                            sum_of_final_term += U[:, f]
                    friends_pred = (1 - alpha) * friends_rating
                    pred = friends_pred + latent_pred
                    sum += derivative_sigmoid(pred)* (sigmoid(pred)-ratingMatrix[u][i])*(alpha*user_latent + (1-alpha)*sum_of_final_term)
            derivative_Vj = sum + item_latent
            old_V[i,:] = item_latent - theta*derivative_Vj

        U = old_U
        V = old_V

        MAE = np.sum(np.absolute(np.subtract(ratingMatrix,np.transpose(U).dot(np.transpose(V)))))/(num_of_users*num_of_items)
        RMSE = np.sqrt(np.sum(np.power(np.subtract(ratingMatrix,np.transpose(U).dot(np.transpose(V))), 2))/(num_of_users*num_of_items))

        print "iteration: " + str(iter) + " MAE: " + str(MAE)
        print "iteration: " + str(iter) +" RMSE: " + str(RMSE)
        log.write("iteration: " + str(iter) +" MAE: " + str(MAE))
        log.write("iteration: " + str(iter) +" RMSE: " + str(RMSE))

    log.close()
    # return [MAE, RMSE]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def frobenius_norm(matrix, row, column):
    sum = 0
    for i in range(0,row-1):
        for j in range(0, column-1):
            sum += math.pow(matrix[i][j],2)
    return sum


process_data("./Yelp_ALL/users.txt", "./Yelp_ALL/items.txt", "./Yelp_ALL/ratings.txt")