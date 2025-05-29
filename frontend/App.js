import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider ,Outlet} from "react-router-dom";
import AboutUs from "./src/components/AboutUs";
import Header from "./src/components/Header";
import Body from "./src/components/Body";
import Service from "./src/components/Services";
import ContactUs from "./src/components/ContactUs";
import Error from "./src/components/Error";
import ToDoList from "./src/components/ToDoList";
import VoiceBot from "./src/components/VoiceBot";
import DiaryPage from "./src/components/DiaryPage";
import LoginRegisterPage from "./src/components/LoginRegisterPage";
const AppLayout = () =>{
   
       return (
       
           <div className="app">
              <Header/>
              <Outlet/>
        </div>
      
       );
   };


   const appRouter = createBrowserRouter([{
    path:"/",
    element:<AppLayout/>,
    children:[{
        path:"/",
        element:<Body/>
    },
        {
        path:"/AboutUs",
        element:<AboutUs/>
    },
    {
        path:"/ContactUs",
        element:<ContactUs/>
    },
    {
        path:"/Services",
        element:<Service/>
    }, 
    {
        path:"/todo",
        element:<ToDoList/>
    }, 
    {
        path:"/voicebot",
        element:<VoiceBot/>
    },
    {
        path:"/diary",
        element:<DiaryPage/>
    },
    {
        path:"/login",
        element:<LoginRegisterPage/>
    },
   ],
   errorElement:<Error/>,

},
]);







    const root = ReactDOM.createRoot(document.getElementById("root"));
    root.render(<RouterProvider router={appRouter}/>);
    