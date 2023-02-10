import { ColorModeContext, useMode } from "./theme";
import { CssBaseline, ThemeProvider } from "@mui/material";
import { Routes, Route } from "react-router-dom";
import React from 'react';
import Topbar from "./scenes/global/Topbar";
import Sidebar from "./scenes/global/Sidebar";
import Dashboard from "./scenes/dashboard";
import Team from "./scenes/team";
import Collection from "./scenes/collection";
import FAQ from "./scenes/faq";
import Checking from "./scenes/checking";
import Posts from "./scenes/posts";
import Comment from "./scenes/comment";
import Reward from "./scenes/reward";
import Test from "./scenes/test";
// import Form from "./scenes/form";
// import Line from "./scenes/dashboard";
// import Pie from "./scenes/pie";
// import FAQ from "./scenes/faq";
// import Geography from "./scenes/geography";
// import Calender from "./scenes/calendar";


const App = () => {
  const [theme, colorMode] = useMode();

  return (
    <ColorModeContext.Provider value={colorMode}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <div className='app'>
          <Sidebar />
          <main className='content'>
            <Topbar />
            
            <Routes>
              <Route path="/" element={<Dashboard />}/>
              <Route path="/team" element={<Team />}/>
              <Route path="/faq" element={<FAQ />}/>
              <Route path="/collect_data" element={<Collection />}/>
              <Route path="/check_data" element={<Checking />}/>
              <Route path="/posts" element={<Posts />}/>
              <Route path="/comment" element={<Comment />}/>
              <Route path="/reward_data" element={<Reward />}/>
              <Route path="/test" element={<Test />}/>
              {/* <Route path="/geography" element={<Geography />}/> */}
              {/* <Route path="/calendar" element={<Calender />}/> */}
            </Routes>

          </main>
        </div>
      </ThemeProvider>
    </ColorModeContext.Provider>
  )
}

export default App