import React from "react";
import {PiLineVerticalBold} from "react-icons/pi";

function NotFoundPage() {
    return (
        <div className="flex w-full px-0 mx-0 text-center text-muted bg-black items-center min-h-[calc(100vh-168px)] justify-center">
            <div className="flex font-bold text-2xl text-center align-middle">
                <h1>404</h1>
                <PiLineVerticalBold className={"my-1"}/>
                <h1>Page Not Found</h1>
            </div>
        </div>
    )
}

export default NotFoundPage