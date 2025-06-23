
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import  RunnableParallel


model1 = ChatOllama(model='gemma3:1b')
model2 = ChatOllama(model='llama3.2')


prompt1 = PromptTemplate(
    template='Generate a short and simple text from the following text\n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate a 5 short question answers from the following text\n {text}',
    input_variables=['text']
)


prompt3 = PromptTemplate(
    template='Merge the provided notes and question answer into a single document\n notes -> {notes}\n -> {quiz}',
    input_variables=['notes', 'quiz']
)


parser = StrOutputParser()


parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})


merge_chain = prompt3 | model1 | parser


chain = parallel_chain | merge_chain


text = """
LinearSVC
class sklearn.svm.LinearSVC(penalty='l2', loss='squared_hinge', *, dual='auto', tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)[source]
Linear Support Vector Classification.

Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.

The main differences between LinearSVC and SVC lie in the loss function used by default, and in the handling of intercept regularization between those two implementations.

This class supports both dense and sparse input and the multiclass support is handled according to a one-vs-the-rest scheme.

Read more in the User Guide.

Parameters:
penalty{‘l1’, ‘l2’}, default=’l2’
Specifies the norm used in the penalization. The ‘l2’ penalty is the standard used in SVC. The ‘l1’ leads to coef_ vectors that are sparse.

loss{‘hinge’, ‘squared_hinge’}, default=’squared_hinge’
Specifies the loss function. ‘hinge’ is the standard SVM loss (used e.g. by the SVC class) while ‘squared_hinge’ is the square of the hinge loss. The combination of penalty='l1' and loss='hinge' is not supported.

dual“auto” or bool, default=”auto”
Select the algorithm to either solve the dual or primal optimization problem. Prefer dual=False when n_samples > n_features. dual="auto" will choose the value of the parameter automatically, based on the values of n_samples, n_features, loss, multi_class and penalty. If n_samples < n_features and optimizer supports chosen loss, multi_class and penalty, then dual will be set to True, otherwise it will be set to False.

Changed in version 1.3: The "auto" option is added in version 1.3 and will be the default in version 1.5.

tolfloat, default=1e-4
Tolerance for stopping criteria.

Cfloat, default=1.0
Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. For an intuitive visualization of the effects of scaling the regularization parameter C, see Scaling the regularization parameter for SVCs.

multi_class{‘ovr’, ‘crammer_singer’}, default=’ovr’
Determines the multi-class strategy if y contains more than two classes. "ovr" trains n_classes one-vs-rest classifiers, while "crammer_singer" optimizes a joint objective over all classes. While crammer_singer is interesting from a theoretical perspective as it is consistent, it is seldom used in practice as it rarely leads to better accuracy and is more expensive to compute. If "crammer_singer" is chosen, the options loss, penalty and dual will be ignored.

fit_interceptbool, default=True
Whether or not to fit an intercept. If set to True, the feature vector is extended to include an intercept term: [x_1, ..., x_n, 1], where 1 corresponds to the intercept. If set to False, no intercept will be used in calculations (i.e. data is expected to be already centered).

intercept_scalingfloat, default=1.0
When fit_intercept is True, the instance vector x becomes [x_1, ..., x_n, intercept_scaling], i.e. a “synthetic” feature with a constant value equal to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic feature weight. Note that liblinear internally penalizes the intercept, treating it like any other term in the feature vector. To reduce the impact of the regularization on the intercept, the intercept_scaling parameter can be set to a value greater than 1; the higher the value of intercept_scaling, the lower the impact of regularization on it. Then, the weights become [w_x_1, ..., w_x_n, w_intercept*intercept_scaling], where w_x_1, ..., w_x_n represent the feature weights and the intercept weight is scaled by intercept_scaling. This scaling allows the intercept term to have a different regularization behavior compared to the other features.

class_weightdict or ‘balanced’, default=None
Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one. The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).

verboseint, default=0
Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in liblinear that, if enabled, may not work properly in a multithreaded context.

random_stateint, RandomState instance or None, default=None
Controls the pseudo random number generation for shuffling the data for the dual coordinate descent (if dual=True). When dual=False the underlying implementation of LinearSVC is not random and random_state has no effect on the results. Pass an int for reproducible output across multiple function calls. See Glossary.

max_iterint, default=1000
The maximum number of iterations to be run.

Attributes:
coef_ndarray of shape (1, n_features) if n_classes == 2 else (n_classes, n_features)
Weights assigned to the features (coefficients in the primal problem).

coef_ is a readonly property derived from raw_coef_ that follows the internal memory layout of liblinear.

intercept_ndarray of shape (1,) if n_classes == 2 else (n_classes,)
Constants in decision function.

classes_ndarray of shape (n_classes,)
The unique classes labels.

n_features_in_int
Number of features seen during fit.

Added in version 0.24.

feature_names_in_ndarray of shape (n_features_in_,)
Names of features seen during fit. Defined only when X has feature names that are all strings.

Added in version 1.0.

n_iter_int
Maximum number of iterations run across all classes.
"""

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()

