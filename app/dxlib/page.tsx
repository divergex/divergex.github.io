import React from 'react';

const DocsPage = () => {
    return (
        <div style={{height: '100vh', overflow: 'hidden'}}>
            <iframe
                src="https://divergex.github.io/dxlib/"
                style={{
                    width: '100%',
                    height: '100%',
                    border: 'none',
                }}
                title="Python Documentation"
            />
        </div>
    )
        ;
};

export default DocsPage;
