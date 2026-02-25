try:
    from qiskit_aer import Aer
except ImportError:
    from qiskit import Aer
from sklearn.metrics import accuracy_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel

def train_quantum_model(X_train, X_test, y_train, y_test):

    feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2)
    backend = Aer.get_backend('qasm_simulator')

    quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=backend)

    qsvc = QSVC(quantum_kernel=quantum_kernel)
    qsvc.fit(X_train[:100], y_train[:100])  # Small subset for quantum

    preds = qsvc.predict(X_test[:50])
    acc = accuracy_score(y_test[:50], preds)

    return acc