% Belief Rule Base Optimization using Anlytical Method Implementation using MATLAB

% Clear Workspace
clear;

% Initial Belief Degrees
x0 = generate_belief_degree(3,3);  % N = 3, L = 3 (Disjunctive BRB. So, L = N)
disp("Initial Belief Degrees = ")
disp(x0)


% Set nondefault solver options
options = optimoptions('fmincon','PlotFcn','optimplotfvalconstr');

% Solve
[solution,objectiveValue] = fmincon(@objectiveFcn,x0,[],[],[],[],...
    zeros(size(x0)),ones(size(x0)),[],options);

% Clear variables
clearvars options

% Display Optimized Values
disp("Least Mean Square Error = ") 
disp(objectiveValue)
disp("Optimized Belief Degrees = ") 
disp(solution)

function f = objectiveFcn(optimInput)
% Training Dataset

M = 12; % No of Inputs-Output Pair
T = 4; % No of Attributes
N = 3; % No of referencial values
L = N; % Number of rules
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


% Define the variables
belief_degrees = optimInput;

ref_val = [1.0 0.5 0.0]; % Utility Scores of H = 1.0, M = 0.5, L = 0.0

% Initialize calculation storing variables
calculated_output = zeros(M,1); 
differences = zeros(M,1);

for i = 1:M
    weights = get_rule_weights(train_input,i,T,N,ref_val); % Rule Weights

    % Calculate Aggregated Belief Degree and Compute Y
    aggregated_belief_degree = calc_aggregated_belief_degree(weights, belief_degrees, N, L);

    calculated_output(i,1) = calculateY(aggregated_belief_degree,ref_val,N);
    differences(i,1) = calculated_output(i,1) - train_output(i,1);
end

% Define Objective Function
f = sum((differences).^2) / M;
end

function arr = generate_belief_degree(N, L)
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

function arr = get_rule_weights(train_input,input_no,no_of_attributes,no_of_ref_val, ref_vals)
    % Input Transformation
    transformed_input = transform_input(train_input(input_no,:),no_of_attributes, no_of_ref_val, ref_vals);
    
    % Rule Activation Weight Calculation    
    matching_degrees = calc_matching_degrees(transformed_input, no_of_attributes, no_of_ref_val); % Calculate Matching Degree    
    combined_matching_degree = calc_combined_matching_degrees(matching_degrees,no_of_ref_val); % Calculate Combined Matching Degree    
    arr = (matching_degrees) ./ (combined_matching_degree); % Calculate Activation Weight
end

function arr = transform_input (input,no_of_attr,no_of_ref_val,ref_vals)
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
    arr = zeros(no_of_ref_val,1);
    for i = 1:no_of_ref_val
        for j = 1:no_of_attributes
            arr(i,1) = arr(i,1) + individual_matching_degree(j,i);
        end
        
    end
end

function val = calc_combined_matching_degrees(matching_degrees, no_of_rules)
    val = 0;
    for i = 1:no_of_rules
       val = val + matching_degrees(i,1);        
    end
end

function arr = calc_aggregated_belief_degree(activation_weight, belief_degree, no_of_ref_val, no_of_rules)
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

function val = calculateY(agg_bel_val, ref_vals,no_ref_val)
    val = 0;
    for i = 1: no_ref_val
        val = val + (agg_bel_val(i,1)*ref_vals(1,i));
    end
end
