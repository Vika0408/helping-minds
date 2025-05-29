import React, { useState } from "react";

const LoginRegisterPage = () => {
  const [isRegister, setIsRegister] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const toggleForm = () => {
    setIsRegister(!isRegister);
    setEmail("");
    setPassword("");
  };

  const handleAuth = () => {
    if (isRegister) {
      // Register logic
      const users = JSON.parse(localStorage.getItem("users") || "[]");
      const exists = users.find((u) => u.email === email);
      if (exists){
          return alert("User already exists!");
      }
      users.push({ email, password });
      localStorage.setItem("users", JSON.stringify(users));
      alert("Registered successfully!");
      setIsRegister(false);
    } else {
      // Login logic
      const users = JSON.parse(localStorage.getItem("users") || "[]");
      const user = users.find((u) => u.email === email && u.password === password);
      if (user) {
        alert("Login successful!");
        // Navigate or set auth state
      } else {
        alert("Invalid credentials!");
      }
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-r from-blue-300 to-purple-400">
      <div className="bg-white p-8 rounded-xl shadow-2xl w-full max-w-md">
        <h2 className="text-3xl font-bold mb-6 text-center text-blue-700">
          {isRegister ? "Register" : "Login"}
        </h2>

        <input
          type="email"
          placeholder="Email"
          className="w-full px-4 py-2 mb-4 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />

        <input
          type="password"
          placeholder="Password"
          className="w-full px-4 py-2 mb-4 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        <button
          onClick={handleAuth}
          className="w-full bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700 transition"
        >
          {isRegister ? "Register" : "Login"}
        </button>

        <p className="mt-4 text-center text-gray-600">
          {isRegister ? "Already have an account?" : "Don't have an account?"}{" "}
          <span
            className="text-blue-700 font-semibold cursor-pointer hover:underline"
            onClick={toggleForm}
          >
            {isRegister ? "Login here" : "Register here"}
          </span>
        </p>
      </div>
    </div>
  );
};

export default LoginRegisterPage;
