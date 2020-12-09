% Belief Rule Base Optimization using Anlytical Method Implementation using MATLAB

% Clear Workspace
clear;

% Training Input
% No of Inputs-Output Pair, M = 12
% No of Attributes, T = 4

M = 12;
T = 4;
train_input = [0.98 1.0 1.0 1.0;
    0.98 0.8 0.8 0.8;
    0.98 0.8 0.2 0.8;
    0.98 0.4 0.4 0.8;
    0.98 0.4 0.6 0.8;
    0.98 1.0 0.0 0.8;
    0.98 0.0 0.0 0.0;
    0.98 1.0 1.0 0.2;
    0.98 0.4 0.6 0.2;
    0.98 0.6 0.4 0.2;
    0.98 0.8 0.2 0.2;
    0.98 0.8 0.8 0.2];

train_output = [1.0;
    0.9;
    0.6;
    0.3;
    0.4;
    0.2;
    0.0;
    0.3;
    0.1;
    0.2;
    0.2;
    0.4];

% No of referencial values, N = 3
N = 3;
ref_values = [1.0 0.5 0.0] % Utility Scores of H = 1.0, M = 0.5, L = 0.0

% Initial Belief Degrees
% Following Disjunctive BRB Approach, 
% In Disjunctive BRB, number of rules, L = Number of referencial values, N
L = N;
initial_belief_degrees = generate_belief_degree(N,L)

% Input to Process
input_number = 3; % This value determines the number of input_output pair we are considering

% Step 01: Input Transformation
% train_input(input_number,:) takes values of the (input_number)th row
transformed_input = transform_input(train_input(input_number,:),T, N, ref_values)

% Step 02: Rule Activation Weight Calculation

% Calculate Matching Degree
matching_degrees = calc_matching_degrees(transformed_input, T, N)
% Calculate Combined Matching Degree
combined_matching_degree = calc_combined_matching_degrees(matching_degrees,L);
% Calculate Activation Weight
activation_weight = (matching_degrees) ./ (combined_matching_degree)

% Step 03: Belief Degree Update

belief_update_factor = 1; % Use this value to update the belief degrees
final_belief_degree = (initial_belief_degrees) .* (belief_update_factor)

% Step 04: Rule Aggregation

% We are using "Analytical Method" of ER on Disjunctive BRB
aggregated_belief_degree = calc_aggregated_belief_degree(activation_weight, final_belief_degree, N, L)

function arr = generate_belief_degree(N, L)
% Function to generate belief degree
    belief_generator = rand(N,L);
    temp_gen_col_total = zeros(L,1);
    arr = zeros(N,L);
    for col = 1:L
        for row = 1:N
            temp_gen_col_total(col,1) = temp_gen_col_total(col,1) + belief_generator(row, col);
        end
    end 

    for row = 1:N
        for col = 1:L
            arr(row,col)  = belief_generator(row,col) ./ temp_gen_col_total(col,1);
        end 
    end
end




function arr = transform_input (input,no_of_attr,no_of_ref_val,ref_vals)
% Input Transformation Function
    
    arr = zeros(no_of_attr,no_of_ref_val); % Initialize with row_number x column_number dummy values
    % Calculate and Populate with original values
    for i = 1:no_of_attr
        if (input(1,i)>= ref_vals(1,2) && input(1,i) <= ref_vals(1,1))
            arr(i,2) = (ref_vals(1,1) - input(1,i))/(ref_vals(1,1) - ref_vals(1,2));
            arr(i,1) = 1 - arr(i,2);
            arr(i,3) = 1 - (arr(i,2) + arr(i,1));
        elseif (input(1,i)>= ref_vals(1,3) && input(1,i) <= ref_vals(1,2))
            arr(i,3) = (ref_vals(1,2) - input(1,i))/(ref_vals(1,2) - ref_vals(1,3));
            arr(i,2) = 1 - arr(i,3);
            arr(i,1) = 1 - (arr(i,2) + arr(i,3));
            
        end
    end
    
end




function arr = calc_matching_degrees(individual_matching_degree, no_of_attributes, no_of_ref_val)
% Function for Calculating Matching Degrees
    arr = zeros(no_of_ref_val,1);
    for i = 1:no_of_ref_val
        for j = 1:no_of_attributes
            arr(i,1) = arr(i,1) + individual_matching_degree(j,i);
        end
        
    end
end




function val = calc_combined_matching_degrees(matching_degrees, no_of_rules)
% Function for Calculating Combined Matching Degree
    val = 0;
    for i = 1:no_of_rules
       val = val + matching_degrees(i,1);        
    end
end




function arr = calc_aggregated_belief_degree(activation_weight, belief_degree, no_of_ref_val, no_of_rules)
% Function for Aggregated Belief Degree Calculation Using Analytical Method
    
    arr = zeros(no_of_ref_val,1);

    partA = calc_Part_A(activation_weight, belief_degree, no_of_ref_val, no_of_rules);
    partB = calc_Part_B(activation_weight, belief_degree, no_of_ref_val, no_of_rules);
    partC = calc_Part_C(activation_weight, no_of_rules);
    
    combined_partA = 0;
    for i = 1:no_of_rules
        combined_partA = combined_partA + partA(i,1);
    end
    
    for j = 1:no_of_ref_val
        arr(j,1) = (partA(j,1) - partB)/((combined_partA - ((no_of_ref_val - 1) * partB)) - partC);
    end    
end



function arr = calc_Part_A(activation_weight, belief_degree, no_of_ref_val, no_of_rules)
    arr = zeros(3,1);
    for i = 1:no_of_ref_val
        for j = 1:no_of_rules
            part1 = activation_weight(i,1) * belief_degree(i,j);
            
            temp = 0;
            for k = 1:no_of_ref_val
                temp = temp + belief_degree(k,j);
            end
            part2 = (1 - (activation_weight(i,1)*temp));
            
            arr(i,1) = part1 + part2;
        end
    end
end



function val = calc_Part_B(activation_weight, belief_degree, no_of_ref_val, no_of_rules)
    val = 1;
    for i = 1:no_of_rules
        
        temp_total_belief = 0;
        
        for j = 1:no_of_ref_val
            temp_total_belief = temp_total_belief + belief_degree(j,i);
        end
        
        temp = activation_weight(i,1) * temp_total_belief;
        val = val * (1 - temp);
    end
end



function val = calc_Part_C(activation_weight, no_of_rules)
    val = 1;
    for i = 1:no_of_rules
        val = val * (1 - activation_weight(i,1));
    end
end
